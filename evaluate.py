

import argparse
import math
from datetime import datetime
import numpy as np
import tensorflow as tf
import socket
import importlib
import os
import sys
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
import pickle

parser = argparse.ArgumentParser()
parser.add_argument('--gpu', type=int, default=0, help='GPU to use [default: GPU 0]')
parser.add_argument('--model', default='model_concat_upsa', help='Model name [default: model_concat_upsa]')
parser.add_argument('--dataset', default='flying_things_dataset', help='Dataset name [default: flying_things_dataset]')
parser.add_argument('--data', default='data_preprocessing/data_processed_maxcut_35_20k_2k_8192', help='Dataset directory [default: /data_preprocessing/data_processed_maxcut_35_20k_2k_8192]')
parser.add_argument('--model_path', default='log_train/model.ckpt', help='model checkpoint file path [default: log_train/model.ckpt]')
parser.add_argument('--log_dir', default='log_evaluate', help='Log dir [default: log_evaluate]')
parser.add_argument('--num_point', type=int, default=2048, help='Point Number [default: 2048]')
parser.add_argument('--batch_size', type=int, default=16, help='Batch Size during training [default: 16]')
FLAGS = parser.parse_args()

os.environ['CUDA_VISIBLE_DEVICES'] = str(FLAGS.gpu)

BATCH_SIZE = FLAGS.batch_size
NUM_POINT = FLAGS.num_point
DATA = FLAGS.data
GPU_INDEX = FLAGS.gpu

MODEL = importlib.import_module(FLAGS.model) # import network module
MODEL_FILE = os.path.join(BASE_DIR, FLAGS.model+'.py')
MODEL_PATH = FLAGS.model_path
LOG_DIR = FLAGS.log_dir
if not os.path.exists(LOG_DIR): os.mkdir(LOG_DIR)
os.system('cp %s %s' % (MODEL_FILE, LOG_DIR)) # bkp of model def
os.system('cp %s %s' % (__file__, LOG_DIR)) # bkp of train procedure
LOG_FOUT = open(os.path.join(LOG_DIR, 'log_evaluate.txt'), 'w')
LOG_FOUT.write(str(FLAGS)+'\n')

DATASET = importlib.import_module(FLAGS.dataset)
TEST_DATASET = DATASET.SceneflowDataset(DATA, npoints=NUM_POINT, train=False)

def log_string(out_str):
    LOG_FOUT.write(out_str+'\n')
    LOG_FOUT.flush()
    print(out_str)

def evaluate():
    with tf.Graph().as_default():
        with tf.device('/gpu:'+str(GPU_INDEX)):
            pointclouds_pl, labels_pl, masks_pl = MODEL.placeholder_inputs(BATCH_SIZE, NUM_POINT)
            is_training_pl = tf.placeholder(tf.bool, shape=())

            print("--- Get model and loss")
            # Get model and loss
            pred, end_points = MODEL.get_model(pointclouds_pl, is_training_pl, bn_decay=None)
            loss = MODEL.get_loss(pred, labels_pl, masks_pl, end_points)
            tf.summary.scalar('loss', loss)

            # Add ops to save and restore all the variables.
            saver = tf.train.Saver()

        # Create a session
        config = tf.ConfigProto()
        config.gpu_options.allow_growth = True
        config.allow_soft_placement = True
        config.log_device_placement = False
        sess = tf.Session(config=config)

        saver.restore(sess, MODEL_PATH)
        log_string("Model restored.")

        ops = {'pointclouds_pl': pointclouds_pl,
               'labels_pl': labels_pl,
               'masks_pl': masks_pl,
               'is_training_pl': is_training_pl,
               'pred': pred,
               'loss': loss}

        eval_one_epoch(sess, ops)

def get_batch(dataset, idxs, start_idx, end_idx):
    bsize = end_idx-start_idx
    batch_data = np.zeros((bsize, NUM_POINT*2, 6))
    batch_label = np.zeros((bsize, NUM_POINT, 3))
    batch_mask = np.zeros((bsize, NUM_POINT))
    # shuffle idx to change point order (change FPS behavior)
    shuffle_idx = np.arange(NUM_POINT)
    np.random.shuffle(shuffle_idx)
    for i in range(bsize):
        pc1, pc2, color1, color2, vel, mask1 = dataset[idxs[i+start_idx]]

        batch_data[i,:NUM_POINT,:3] = pc1[shuffle_idx,:]
        batch_data[i,:NUM_POINT,3:] = color1[shuffle_idx,:]
        batch_data[i,NUM_POINT:,:3] = pc2[shuffle_idx,:]
        batch_data[i,NUM_POINT:,3:] = color2[shuffle_idx,:]
        batch_label[i,:,:] = vel[shuffle_idx,:]
        batch_mask[i,:] = mask1[shuffle_idx]
    return batch_data, batch_label, batch_mask

def scene_flow_EPE_np(pred, labels, mask):
    error = np.sqrt(np.sum((pred - labels)**2, 2) + 1e-20)

    gtflow_len = np.sqrt(np.sum(labels*labels, 2) + 1e-20) # B,N
    acc1 = np.sum(np.logical_or((error <= 0.05)*mask, (error/gtflow_len <= 0.05)*mask), axis=1)
    acc2 = np.sum(np.logical_or((error <= 0.1)*mask, (error/gtflow_len <= 0.1)*mask), axis=1)

    mask_sum = np.sum(mask, 1)
    acc1 = acc1[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc1 = np.mean(acc1)
    acc2 = acc2[mask_sum > 0] / mask_sum[mask_sum > 0]
    acc2 = np.mean(acc2)

    EPE = np.sum(error * mask, 1)[mask_sum > 0] / mask_sum[mask_sum > 0]
    EPE = np.mean(EPE)
    return EPE, acc1, acc2

def eval_one_epoch(sess, ops):
    """ ops: dict mapping from string to tf ops """
    is_training = False
    test_idxs = np.arange(0, len(TEST_DATASET))
    # Test on all data: last batch might be smaller than BATCH_SIZE
    num_batches = (len(TEST_DATASET)+BATCH_SIZE-1) // BATCH_SIZE

    loss_sum = 0
    epe_3d_sum = 0
    acc_3d_sum = 0
    acc_3d_2_sum = 0

    log_string(str(datetime.now()))
    log_string('---- EVALUATION ----')

    batch_data = np.zeros((BATCH_SIZE, NUM_POINT*2, 3))
    batch_label = np.zeros((BATCH_SIZE, NUM_POINT, 3))
    batch_mask = np.zeros((BATCH_SIZE, NUM_POINT))
    for batch_idx in range(num_batches):
        if batch_idx %20==0:
            log_string('%03d/%03d'%(batch_idx, num_batches))
        start_idx = batch_idx * BATCH_SIZE
        end_idx = min(len(TEST_DATASET), (batch_idx+1) * BATCH_SIZE)
        cur_batch_size = end_idx-start_idx
        cur_batch_data, cur_batch_label, cur_batch_mask = get_batch(TEST_DATASET, test_idxs, start_idx, end_idx)
        if cur_batch_size == BATCH_SIZE:
            batch_data = cur_batch_data
            batch_label = cur_batch_label
            batch_mask = cur_batch_mask
        else:
            batch_data[0:cur_batch_size] = cur_batch_data
            batch_label[0:cur_batch_size] = cur_batch_label
            batch_mask[0:cur_batch_size] = cur_batch_mask

        # ---------------------------------------------------------------------
        # ---- INFERENCE BELOW ----
        pred_val_sum = np.zeros((BATCH_SIZE, NUM_POINT, 3))
        SHUFFLE_TIMES = 10
        RECURRENT_TIMES = 0
        for shuffle_cnt in range(SHUFFLE_TIMES):
            shuffle_idx = np.arange(NUM_POINT)
            np.random.shuffle(shuffle_idx)
            batch_data_new = np.copy(batch_data)
            batch_data_new[:,0:NUM_POINT,:] = batch_data[:,shuffle_idx,:]
            batch_data_new[:,NUM_POINT:,:] = batch_data[:,NUM_POINT+shuffle_idx,:]
            feed_dict = {ops['pointclouds_pl']: batch_data_new,
                         ops['labels_pl']: batch_label[:,shuffle_idx,:],
                         ops['masks_pl']: batch_mask[:,shuffle_idx],
                         ops['is_training_pl']: is_training}
            loss_val, pred_val = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
            for recurrent_cnt in range(RECURRENT_TIMES):
                batch_data_new[:,0:NUM_POINT,0:3] += pred_val
                batch_label_new = np.copy(batch_label)
                batch_label_new[:,:,:] = batch_label - pred_val
                feed_dict = {ops['pointclouds_pl']: batch_data_new,
                             ops['labels_pl']: batch_label_new[:,shuffle_idx,:],
                             ops['masks_pl']: batch_mask[:,shuffle_idx],
                             ops['is_training_pl']: is_training}
                loss_val, pred_val_new = sess.run([ops['loss'], ops['pred']], feed_dict=feed_dict)
                pred_val += pred_val_new
            pred_val_sum[:,shuffle_idx,:] += pred_val
        # ---- INFERENCE ABOVE ----
        # ---------------------------------------------------------------------
        pred_val = pred_val_sum / float(SHUFFLE_TIMES)
        tmp = np.sum((pred_val - batch_label)**2, 2) / 2.0
        loss_val_np = np.mean(batch_mask * tmp)
        loss_val = loss_val_np
        print('batch loss: %f' % (loss_val))
        if cur_batch_size==BATCH_SIZE:
            loss_sum += loss_val

        epe_3d, acc_3d, acc_3d_2 = scene_flow_EPE_np(pred_val, batch_label, batch_mask)
        print('batch EPE 3D: %f\tACC 3D: %f\tACC 3D 2: %f' % (epe_3d, acc_3d, acc_3d_2))

        if cur_batch_size==BATCH_SIZE:
            epe_3d_sum += epe_3d
            acc_3d_sum += acc_3d
            acc_3d_2_sum += acc_3d_2

    log_string('eval mean loss: %f' % (loss_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval mean EPE 3D: %f' % (epe_3d_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval mean ACC 3D: %f' % (acc_3d_sum / float(len(TEST_DATASET)/BATCH_SIZE)))
    log_string('eval mean ACC 3D 2: %f' % (acc_3d_2_sum / float(len(TEST_DATASET)/BATCH_SIZE)))

    return loss_sum/float(len(TEST_DATASET)/BATCH_SIZE)


if __name__ == "__main__":
    log_string('pid: %s'%(str(os.getpid())))
    evaluate()
    LOG_FOUT.close()
