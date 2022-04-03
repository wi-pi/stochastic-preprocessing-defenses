import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import tensorflow as tf
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)
from cleverhans.dataset import CIFAR10
from cleverhans.evaluation import batch_eval
from cleverhans.model_zoo.madry_lab_challenges.cifar10_model import make_wresnet as ResNet
from cleverhans.utils_tf import initialize_uninitialized_global_variables
from tqdm import trange

ext_path = Path(__file__).parent / 'icml19_public'
sys.path.append(str(ext_path))

from .utils import do_eval, init_defense


# noinspection DuplicatedCode
def parse_args():
    parser = argparse.ArgumentParser()
    # basic
    parser.add_argument('-b', '--batch', type=int, default=500)
    parser.add_argument('-g', '--gpu', type=str)
    # attack
    parser.add_argument('--eps', type=float, default=8)
    parser.add_argument('--lr', type=float, default=1)
    parser.add_argument('--step', type=int, default=10)
    parser.add_argument('--eot', type=int, default=1)
    # model & dataset
    parser.add_argument('--model-dir', type=str, default='static/models/models/naturally_trained')
    parser.add_argument('--test-size', type=int, default=500)

    args = parser.parse_args()
    return args


# noinspection DuplicatedCode
def main(args):
    # Basic
    if args.gpu:
        os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu

    """Load data
    """
    data = CIFAR10()
    x_test, y_test = data.get_set('test')
    indices = np.random.default_rng(0).permutation(len(x_test))[:args.test_size]
    x_test = x_test[indices] * 255
    y_test = y_test[indices]

    img_rows, img_cols, nchannels = x_test.shape[1:4]
    nb_classes = y_test.shape[1]

    """Load base model
    """
    sess = tf.Session()
    x = tf.placeholder(tf.float32, shape=(None, img_rows, img_cols, nchannels))
    y = tf.placeholder(tf.float32, shape=(None, nb_classes))
    model = ResNet(scope='ResNet')
    preds = model.get_logits(x)

    ckpt_path = tf.train.get_checkpoint_state(args.model_dir).model_checkpoint_path
    saver = tf.train.Saver(var_list=dict((v.name.split('/', 1)[1].split(':')[0], v) for v in tf.global_variables()))
    saver.restore(sess, ckpt_path)
    initialize_uninitialized_global_variables(sess)

    """Initialize the defense
    """
    multi_noise = False
    predictor = init_defense(sess, x, preds, args.batch, multi_noise=multi_noise)

    """Evaluate clean
    """
    do_eval(sess, x, y, preds, x_test, y_test, '', False, predictor=predictor, batch_size=args.batch)

    """Generate adv examples
    """
    y_cls = np.argmax(y_test, axis=-1)
    assert y_cls[0] != y_cls[1]
    logits_clean = batch_eval(sess, [x], [preds], [x_test], batch_size=args.batch)[0]
    assert np.argmax(logits_clean[0]) == y_cls[0]
    assert np.argmax(logits_clean[1]) == y_cls[1]
    target_logits = logits_clean.copy()
    target_logits[:] = logits_clean[0]
    target_logits[y_cls == y_cls[0]] = logits_clean[1]

    target_logits_ph = tf.placeholder(tf.float32, shape=(None, nb_classes))
    loss = tf.reduce_sum(tf.square(target_logits_ph - preds))
    grad = tf.gradients(loss, x)[0]

    n_batches = math.ceil(x_test.shape[0] / args.batch)
    X_adv_all2 = x_test.copy()

    for b in trange(n_batches, desc='Attack'):
        X = x_test[b * args.batch:(b + 1) * args.batch]
        Y = y_cls[b * args.batch:(b + 1) * args.batch]
        targets = target_logits[b * args.batch:(b + 1) * args.batch]

        X_adv = X.copy()

        # nb_iter = 100
        # step = (2.5 * eps) / nb_iter
        # nb_rand = 40

        # choose the bound for the EOT noise to match the magnitude of the noise used by the defense
        if multi_noise:
            eps_noise = 0.01 * 255
        else:
            eps_noise = 30.0

        for i in trange(args.step, desc='PGD'):
            # loss_np, grad_np, preds_np = sess.run([loss, grad, preds], feed_dict={x: X_adv, target_logits_ph: targets})
            loss_np, grad_np, preds_np = 0, 0, 0

            for j in range(args.eot):

                # if the defense uses multiple types of noise, perform EOT over all types
                if multi_noise:
                    if j % 2 == 0:
                        noise = np.random.normal(0., 1., size=X_adv.shape)
                    elif j % 2 == 1:
                        noise = np.random.uniform(-1., 1., size=X_adv.shape)
                    else:
                        noise = np.sign(np.random.uniform(-1., 1., size=X_adv.shape))
                else:
                    noise = np.random.normal(0., 1., size=X_adv.shape)

                X_adv_noisy = X_adv + noise * eps_noise
                X_adv_noisy = X_adv_noisy.clip(0, 255)
                loss_npi, grad_npi, preds_npi = sess.run([loss, grad, preds],
                                                         feed_dict={x: X_adv_noisy, target_logits_ph: targets})

                loss_np += loss_npi
                grad_np += grad_npi

            loss_np /= args.eot
            grad_np /= args.eot

            X_adv -= args.lr * np.sign(grad_np)
            X_adv = np.clip(X_adv, X - args.eps, X + args.eps)
            X_adv = np.clip(X_adv, 0, 255)

            # if i % 10 == 0:
            #     print(b, i, loss_np, np.mean(np.argmax(preds_np, axis=-1) == Y))

        X_adv_all2[b * args.batch:(b + 1) * args.batch] = X_adv

    """Evaluate adv examples
    """
    do_eval(sess, x, y, preds, X_adv_all2, y_test, '', True, predictor=predictor, batch_size=args.batch)


if __name__ == '__main__':
    main(parse_args())
