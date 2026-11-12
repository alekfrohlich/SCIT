import numpy as np
import tensorflow as tf
from sklearn.metrics.pairwise import rbf_kernel
import methods.DGCIT.cit_gan as cit_gan
from scipy.stats import rankdata, ks_2samp, wilcoxon
import methods.DGCIT.gan_utils as gan_utils
import pandas as pd
from sklearn.model_selection import KFold
tf.random.set_seed(42)
tf.compat.v1.enable_eager_execution()

np.random.seed(42)

#
# The generate_samples_random function and rdc function were inspired by
# GCIT Github repository by Alexis Bellot1,2 Mihaela van der Schaar
# source: https://github.com/alexisbellot/GCIT
#


def generate_samples_random(size=1000, sType='CI', dx=1, dy=1, dz=20, nstd=0.05, alpha_x=0.05,
                            normalize=True, seed=None, dist_z='gaussian'):
    '''
    Generate CI,I or NI post-nonlinear samples
    1. Z is independent Gaussian or Laplace
    2. X = f1(<a,Z> + b + noise) and Y = f2(<c,Z> + d + noise) in case of CI
    Arguments:
        size : number of samples
        sType: CI, I, or NI
        dx: Dimension of X
        dy: Dimension of Y
        dz: Dimension of Z
        nstd: noise standard deviation
        we set f1 to be sin function and f2 to be cos function.
    Output:
        Samples X, Y, Z
    '''
    if seed is None:
        np.random.seed()
    else:
        np.random.seed(seed)

    num = size

    if dist_z == 'gaussian':
        cov = np.eye(dz)
        mu = np.zeros(dz)
        Z = np.random.multivariate_normal(mu, cov, num)

    elif dist_z == 'laplace':
        Z = np.random.laplace(loc=0.0, scale=1.0, size=num*dz)
        Z = np.reshape(Z, (num, dz))

    Ax = np.random.rand(dz, dx)
    for i in range(dx):
        Ax[:, i] = Ax[:, i] / np.linalg.norm(Ax[:, i], ord=1)
    Ay = np.random.rand(dz, dy)
    for i in range(dy):
        Ay[:, i] = Ay[:, i] / np.linalg.norm(Ay[:, i], ord=1)

    Axy = np.ones((dx, dy)) * alpha_x

    if sType == 'CI':
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    elif sType == 'I':
        X = np.sin(nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(nstd * np.random.multivariate_normal(np.zeros(dy), np.eye(dy), num))
    else:
        X = np.sin(np.matmul(Z, Ax) + nstd * np.random.multivariate_normal(np.zeros(dx), np.eye(dx), num))
        Y = np.cos(np.matmul(X, Axy) + np.matmul(Z, Ay) + nstd * np.random.multivariate_normal(np.zeros(dx),
                                                                                               np.eye(dx), num))

    if normalize:
        Z = (Z - Z.min()) / (Z.max() - Z.min())
        X = (X - X.min()) / (X.max() - X.min())
        Y = (Y - Y.min()) / (Y.max() - Y.min())

    return np.array(X), np.array(Y), np.array(Z)


#
# test statistics for DGCIT
#


def t_and_sigma(psy_x_i, psy_y_i, phi_x_i, phi_y_i):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    t_b = tf.reduce_sum(matrix, axis=1) / tf.cast(n, tf.float64)
    t_b = tf.expand_dims(t_b, axis=1)

    crit_matrix = matrix - t_b
    std_b = tf.sqrt(tf.reduce_sum(crit_matrix**2, axis=1) / tf.cast(n-1, tf.float64))
    return t_b, std_b


def test_statistics(psy_x_i, psy_y_i, phi_x_i, phi_y_i, t_b, std_b, j):
    b, n = psy_x_i.shape
    x_mtx = phi_x_i - psy_x_i
    y_mtx = phi_y_i - psy_y_i
    matrix = tf.reshape(x_mtx[None, :, :] * y_mtx[:, None, :], [-1, n])
    crit_matrix = matrix - t_b
    test_stat = tf.reduce_max(tf.abs(tf.sqrt(tf.cast(n, tf.float64)) * tf.squeeze(t_b) / std_b))

    sig = tf.reduce_sum(crit_matrix[None, :, :] * crit_matrix[:, None, :], axis=2)
    coef = std_b[None, :] * std_b[:, None] * tf.cast(n-1, tf.float64)
    sig_xy = sig / coef

    eigenvalues, eigenvectors = tf.linalg.eigh(sig_xy)
    base = tf.zeros_like(eigenvectors)
    eig_vals = tf.sqrt(eigenvalues + 1e-12)
    lamda = tf.linalg.set_diag(base, eig_vals)
    sig_sqrt = tf.matmul(tf.matmul(eigenvectors, lamda), tf.linalg.inv(eigenvectors))

    z_samples = tf.random.normal([b*b, j ], 0,1)
    z_samples = tf.cast(z_samples, tf.float64)
    vals = tf.matmul(sig_sqrt, z_samples)
    t_j = tf.reduce_max(vals, axis=0)
    return test_stat, t_j

#
# Training algorithm for DGCIT
#


def dgcit(n=500, z_dim=100, simulation='type1error', batch_size=64, n_iter=1000, train_writer=None,
          current_iters=0, nstd=1.0, z_dist='gaussian', x_dims=1, y_dims=1, a_x=0.05, M=500, k=2,
          var_idx=1, b=30, j=1000):
    # generate samples x, y, z
    # arguments: size, sType='CI', dx=1, dy=1, dz=20, nstd=1, fixed_function='linear',
    # debug=False, normalize=True, seed=None, dist_z='gaussian'
    if simulation == 'type1error':
        # generate samples x, y, z under null hypothesis - x and y are conditional independent
        x, y, z = generate_samples_random(size=n, sType='CI', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd, alpha_x=a_x,
                                          dist_z=z_dist)

    elif simulation == 'power':
        # generate samples x, y, z under alternative hypothesis - x and y are dependent
        x, y, z = generate_samples_random(size=n, sType='dependent', dx=x_dims, dy=y_dims, dz=z_dim, nstd=nstd,
                                          alpha_x=a_x, dist_z=z_dist)
    

    p_value = test_dgcit(x, y, z, batch_size=batch_size, n_iter=n_iter,
                         train_writer=train_writer, current_iters=current_iters, M=M, k=k, b=b, j=j)
    

    print("DGCIT p-value:", p_value)
    exit(0)
    return p_value

def test_dgcit(x,y,z, z_dim=100, batch_size=64, n_iter=1000,
               train_writer=None, current_iters=0, M=100, k=2, b=30, j=750, lr=0.0005,
               gen_clipping_val=0.5, gen_clipping_norm=1.0, w_clipping_val=0.5,
               w_clipping_norm=1.0, scaling_coef=1.0, sinkhorn_eps=0.8, sinkhorn_l=30):

    n = x.shape[0]
    z_dim = z.shape[1]
    x_dims = x.shape[1]
    y_dims = y.shape[1]
    # no. of random and hidden dimensions
    v_dims = int(50)
    h_dims = int(512)

    # create instance of G & D
    generator_x = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, x_dims, batch_size)
    generator_y = cit_gan.WGanGenerator(n, z_dim, h_dims, v_dims, y_dims, batch_size)
    discriminator_x = cit_gan.WGanDiscriminator(n, z_dim, h_dims, x_dims, batch_size)
    discriminator_y = cit_gan.WGanDiscriminator(n, z_dim, h_dims, y_dims, batch_size)


    gx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dx_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)
    gy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=gen_clipping_norm, clipvalue=gen_clipping_val)
    dy_optimiser = tf.keras.optimizers.Adam(lr, beta_1=0.5, clipnorm=w_clipping_norm, clipvalue=w_clipping_val)

    @tf.function
    def x_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_x.call(gen_inputs)
        fake_x_p = generator_x.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                             f_real_p, f_fake_p)
            # disc_loss = - tf.math.minimum(loss1, 1)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_x.trainable_variables)
        dx_optimiser.apply_gradients(zip(d_grads, discriminator_x.trainable_variables))

    @tf.function
    def x_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_x.call(gen_inputs)
            fake_x_p = generator_x.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_x.call(d_real)
            f_fake = discriminator_x.call(d_fake)
            f_real_p = discriminator_x.call(d_real_p)
            f_fake_p = discriminator_x.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                                           sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_x.trainable_variables)
        gx_optimiser.apply_gradients(zip(generator_grads, generator_x.trainable_variables))
        return gen_loss

    @tf.function
    def y_update_d(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        fake_x = generator_y.call(gen_inputs)
        fake_x_p = generator_y.call(gen_inputs_p)
        d_fake = tf.concat([fake_x, real_z], axis=1)
        d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)

        with tf.GradientTape() as disc_tape:
            f_real = discriminator_y.call(d_real)
            f_fake = discriminator_y.call(d_fake)
            f_real_p = discriminator_y.call(d_real_p)
            f_fake_p = discriminator_y.call(d_fake_p)
            # call compute loss using @tf.function + autograph

            loss1 = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps, sinkhorn_l,
                                             f_real_p, f_fake_p)
            disc_loss = - loss1
        # update discriminator parameters
        d_grads = disc_tape.gradient(disc_loss, discriminator_y.trainable_variables)
        dy_optimiser.apply_gradients(zip(d_grads, discriminator_y.trainable_variables))

    @tf.function
    def y_update_g(real_x, real_x_p, real_z, real_z_p, v, v_p):
        gen_inputs = tf.concat([real_z, v], axis=1)
        gen_inputs_p = tf.concat([real_z_p, v_p], axis=1)
        # concatenate real inputs for WGAN discriminator (x, z)
        d_real = tf.concat([real_x, real_z], axis=1)
        d_real_p = tf.concat([real_x_p, real_z_p], axis=1)
        with tf.GradientTape() as gen_tape:
            fake_x = generator_y.call(gen_inputs)
            fake_x_p = generator_y.call(gen_inputs_p)
            d_fake = tf.concat([fake_x, real_z], axis=1)
            d_fake_p = tf.concat([fake_x_p, real_z_p], axis=1)
            f_real = discriminator_y.call(d_real)
            f_fake = discriminator_y.call(d_fake)
            f_real_p = discriminator_y.call(d_real_p)
            f_fake_p = discriminator_y.call(d_fake_p)
            # call compute loss using @tf.function + autograph
            gen_loss = gan_utils.benchmark_loss(f_real, f_fake, scaling_coef, sinkhorn_eps,
                                                                           sinkhorn_l, f_real_p, f_fake_p)
        # update generator parameters
        generator_grads = gen_tape.gradient(gen_loss, generator_y.trainable_variables)
        gy_optimiser.apply_gradients(zip(generator_grads, generator_y.trainable_variables))
        return gen_loss

    psy_x_all = []
    phi_x_all = []
    psy_y_all = []
    phi_y_all = []
    test_samples = b
    test_size = int(n/k)

    # split the train-test sets to k folds
    kf = KFold(n_splits=k, shuffle=True, random_state=42)

    epochs = int(n_iter)

    x_orig = x
    y_orig = y
    z_orig = z

    for train_idx, test_idx in kf.split(np.arange(x_orig.shape[0])):
        x_train, y_train, z_train = x_orig[train_idx], y_orig[train_idx], z_orig[train_idx]
        dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train,
                                                      z_train))
        # Repeat n epochs
        training = dataset.repeat(epochs)
        training_dataset = training.shuffle(100).batch(batch_size * 2)
        # test-set is the one left
        testing_dataset = tf.data.Dataset.from_tensor_slices((x_orig[test_idx], y_orig[test_idx], z_orig[test_idx]))

        for x_batch, y_batch, z_batch in training_dataset.take(n_iter):
            if x_batch.shape[0] != batch_size * 2:
                continue

            # seperate the batch into two parts to train two gans
            x_batch1 = x_batch[:batch_size, ...]
            x_batch2 = x_batch[batch_size:, ...]
            y_batch1 = y_batch[:batch_size, ...]
            y_batch2 = y_batch[batch_size:, ...]
            z_batch1 = z_batch[:batch_size, ...]
            z_batch2 = z_batch[batch_size:, ...]

            noise_v = tf.random.normal([batch_size, v_dims], mean=0.0, stddev=tf.sqrt(1.0 / 3.0))
            noise_v = tf.cast(noise_v, tf.float64)
            noise_v_p = tf.random.normal([batch_size, v_dims], mean=0.0, stddev=tf.sqrt(1.0 / 3.0))
            noise_v_p = tf.cast(noise_v_p, tf.float64)
            x_update_d(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
            loss_x = x_update_g(x_batch1, x_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
            y_update_d(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)
            loss_y = y_update_g(y_batch1, y_batch2, z_batch1, z_batch2, noise_v, noise_v_p)


            current_iters += 1

        psy_x_b = []
        phi_x_b = []
        psy_y_b = []
        phi_y_b = []

        x_samples = []
        y_samples = []
        z = []
        x = []
        y = []

        # the following code generate x_1, ..., x_400 for all B and it takes 61 secs for one test
        for test_x, test_y, test_z in testing_dataset:
            test_z_pre_tile = tf.reshape(test_z, [1, z_dim])
            tiled_z = tf.tile(test_z_pre_tile, [M, 1])
            noise_v = tf.random.normal([M, v_dims], mean=0.0, stddev=tf.sqrt(1.0 / 3.0))
            noise_v = tf.cast(noise_v, tf.float64)
            g_inputs = tf.concat([tiled_z, noise_v], axis=1)
            # generator samples from G and evaluate from D
            fake_x = generator_x.call(g_inputs, training=False)
            fake_y = generator_y.call(g_inputs, training=False)
            x_samples.append(fake_x)
            y_samples.append(fake_y)
            z.append(test_z)
            x.append(test_x)
            y.append(test_y)

        standardise = True

        if standardise:
            x_samples = (x_samples - tf.reduce_mean(x_samples)) / tf.math.reduce_std(x_samples)
            y_samples = (y_samples - tf.reduce_mean(y_samples)) / tf.math.reduce_std(y_samples)
            x = (x - tf.reduce_mean(x)) / tf.math.reduce_std(x)
            y = (y - tf.reduce_mean(y)) / tf.math.reduce_std(y)
            z = (z - tf.reduce_mean(z)) / tf.math.reduce_std(z)

        f1 = cit_gan.CharacteristicFunction(M, x_dims, z_dim, test_size)
        f2 = cit_gan.CharacteristicFunction(M, y_dims, z_dim, test_size)
        for i in range(test_samples):
            phi_x = tf.reduce_mean(f1.call(x_samples, z), axis=1)
            phi_y = tf.reduce_mean(f2.call(y_samples, z), axis=1)
            psy_x = tf.squeeze(f1.call(x, z))
            psy_y = tf.squeeze(f2.call(y, z))

            psy_x_b.append(psy_x)
            phi_x_b.append(phi_x)
            psy_y_b.append(psy_y)
            phi_y_b.append(phi_y)
            f1.update()
            f2.update()

        psy_x_all.append(psy_x_b)
        phi_x_all.append(phi_x_b)
        psy_y_all.append(psy_y_b)
        phi_y_all.append(phi_y_b)

    # reshape
    psy_x_all = tf.reshape(psy_x_all, [k, test_samples, test_size])
    psy_y_all = tf.reshape(psy_y_all, [k, test_samples, test_size])
    phi_x_all = tf.reshape(phi_x_all, [k, test_samples, test_size])
    phi_y_all = tf.reshape(phi_y_all, [k, test_samples, test_size])
    
    t_b = 0.0
    std_b = 0.0
    for n in range(k):
        t, std = t_and_sigma(psy_x_all[n], psy_y_all[n], phi_x_all[n], phi_y_all[n])
        t_b += t
        std_b += std
    t_b = t_b / tf.cast(k, tf.float64)
    std_b = std_b / tf.cast(k, tf.float64)

    psy_x_all = tf.transpose(psy_x_all, (1, 0, 2))
    psy_y_all = tf.transpose(psy_y_all, (1, 0, 2))
    phi_x_all = tf.transpose(phi_x_all, (1, 0, 2))
    phi_y_all = tf.transpose(phi_y_all, (1, 0, 2))

    psy_x_all = tf.reshape(psy_x_all, [test_samples, test_size*k])
    psy_y_all = tf.reshape(psy_y_all, [test_samples, test_size*k])
    phi_x_all = tf.reshape(phi_x_all, [test_samples, test_size*k])
    phi_y_all = tf.reshape(phi_y_all, [test_samples, test_size*k])

    stat, critical_vals = test_statistics(psy_x_all, psy_y_all, phi_x_all, phi_y_all, t_b, std_b, j)
    comparison = [c > stat or c == stat for c in critical_vals]
    comparison = np.reshape(comparison, (-1,))
    p_value = np.sum(comparison.astype(np.float32)) / j
    return p_value


def rdc(x, y, f=np.sin, k=20, s=1 / 6., n=1):
    """
    Computes the Randomized Dependence Coefficient
    x,y: numpy arrays 1-D or 2-D
         If 1-D, size (samples,)
         If 2-D, size (samples, variables)
    f:   function to use for random projection
    k:   number of random projections to use
    s:   scale parameter
    n:   number of times to compute the RDC and
         return the median (for stability)
    According to the paper, the coefficient should be relatively insensitive to
    the settings of the f, k, and s parameters.

    Source: https://github.com/garydoranjr/rdc
    """
    x = tf.reshape(x, shape=(x.shape[0], ))
    y = tf.reshape(y, shape=(y.shape[0], ))

    if n > 1:
        values = []
        for i in range(n):
            try:
                values.append(rdc(x, y, f, k, s, 1))
            except np.linalg.linalg.LinAlgError:
                pass
        return np.median(values)

    if len(x.shape) == 1: x = tf.reshape(x, shape=(-1, 1))
    if len(y.shape) == 1: y = tf.reshape(y, shape=(-1, 1))

    # Copula Transformation
    cx = np.column_stack([rankdata(xc, method='ordinal') for xc in np.transpose(x)]) / float(x.shape[0])
    cy = np.column_stack([rankdata(yc, method='ordinal') for yc in np.transpose(y)]) / float(y.shape[0])

    # Add a vector of ones so that w.x + b is just a dot product
    O = np.ones(cx.shape[0])
    X = np.column_stack([cx, O])
    Y = np.column_stack([cy, O])

    # Random linear projections
    Rx = (s / X.shape[1]) * np.random.randn(X.shape[1], k)
    Ry = (s / Y.shape[1]) * np.random.randn(Y.shape[1], k)
    X = np.dot(X, Rx)
    Y = np.dot(Y, Ry)

    # Apply non-linear function to random projections
    fX = f(X)
    fY = f(Y)

    # Compute full covariance matrix
    C = np.cov(np.hstack([fX, fY]).T)

    # Due to numerical issues, if k is too large,
    # then rank(fX) < k or rank(fY) < k, so we need
    # to find the largest k such that the eigenvalues
    # (canonical correlations) are real-valued
    k0 = k
    lb = 1
    ub = k
    while True:
        # Compute canonical correlations
        Cxx = C[:k, :k]
        Cyy = C[k0:k0 + k, k0:k0 + k]
        Cxy = C[:k, k0:k0 + k]
        Cyx = C[k0:k0 + k, :k]

        eigs = np.linalg.eigvals(np.dot(np.dot(np.linalg.pinv(Cxx), Cxy),
                                        np.dot(np.linalg.pinv(Cyy), Cyx)))

        # Binary search if k is too large
        if not (np.all(np.isreal(eigs)) and
                0 <= np.min(eigs) and
                np.max(eigs) <= 1):
            ub -= 1
            k = (ub + lb) // 2
            continue
        if lb == ub: break
        lb = k
        if ub == lb + 1:
            k = ub
        else:
            k = (ub + lb) // 2

    return np.sqrt(np.max(eigs))


def permute(x):
    n = x.shape[0]
    idx = np.random.permutation(n)
    out = x[idx]
    return out

#
# test statistics and GCIT method
# Paper link: https://arxiv.org/pdf/1907.04068.pdf
#


def mmd_squared(X, Y, gamma=1):
    X = X.reshape((len(X)), 1)
    Y = Y.reshape((len(Y)), 1)

    K_XX = rbf_kernel(X, gamma=gamma)
    K_YY = rbf_kernel(Y, gamma=gamma)
    K_XY = rbf_kernel(X, Y, gamma=gamma)

    n = K_XX.shape[0]
    m = K_YY.shape[0]

    mmd_squared = (np.sum(K_XX)-np.trace(K_XX))/(n*(n-1)) + (np.sum(K_YY)-np.trace(K_YY))/(m*(m-1)) - 2 * np.sum(K_XY) / (m * n)

    return mmd_squared


def correlation(X, Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return np.abs(np.corrcoef(X, Y)[0, 1])


def kolmogorov(X, Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return ks_2samp(X, Y)[0]


def wilcox(X, Y):
    X = X.reshape((len(X)))
    Y = Y.reshape((len(Y)))
    return wilcoxon(X, Y)[0]



#
# KCIT experiment results are obtained by running the original implementation.
# Paper link: https://arxiv.org/ftp/arxiv/papers/1202/1202.3775.pdf
#
