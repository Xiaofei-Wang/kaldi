#!/usr/bin/env python

# Copyright 2017 Xiaofei Wang

""" This script computes the MSE for the given feature directory using tdnn repository.
"""

from __future__ import print_function
import numpy as np
import argparse
import logging
import pprint
import os
import sys
import traceback
import kaldi_io
import pickle

sys.path.insert(0, 'steps')
import libs.common as common_lib

logger = logging.getLogger('libs')
logger.setLevel(logging.INFO)
handler = logging.StreamHandler()
handler.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s [%(pathname)s:%(lineno)s - "
                              "%(funcName)s - %(levelname)s ] %(message)s")
handler.setFormatter(formatter)
logger.addHandler(handler)
logger.info('Starting computing the MSE via autoencoder.')

def get_args():
    """ Get args from stdin.

    The common options are defined in the object
    libs.nnet3.train.common.CommonParser.parser.
    See steps/libs/nnet3/train/common.py
    """

    parser = argparse.ArgumentParser(
        description="""This script computes the MSE for the given feature directory using tdnn repository.""",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
        conflict_handler='resolve')

    # egs extraction options
    parser.add_argument("--cmd", type=str, dest="command",
                        action=common_lib.NullstrToNoneAction,
                        help="""Specifies the script to launch jobs.
                        e.g. queue.pl for launching on SGE cluster
                        run.pl for launching on local machine
                        """, default="queue.pl")
    parser.add_argument("--pca-trans-feat-dir", type=str, dest='pca_trans_feat_dir',
                        required=True,
                        help="Directory with features for testing.")
    parser.add_argument("--autoencoder-nnet", dest='autoencoder_nnet', default=None,
                        type=str,
                        action=common_lib.NullstrToNoneAction,
                        help="Directory with trained tdnn auto-encoder.")
    parser.add_argument("--dir", type=str, required=True,
                        help="Directory to store the outputs.")
    parser.add_argument("--cmvn", type=str,
                        action=common_lib.NullstrToNoneAction,
                        default=None,
                        help="Apply CMVN on input features")
    parser.add_argument("--reporting.email", dest="email",
                        type=str, default=None,
                        action=common_lib.NullstrToNoneAction,
                        help="""Email-id to report about the progress 
                        of the experiment. NOTE: It assumes the 
                        machine on which the script is being run can 
                        send emails from command line via. mail 
                        program. The kaldi mailing list will not 
                        support this feature. It might require local 
                        expertise to setup. """)

    print(' '.join(sys.argv))
    print(sys.argv)
    args = parser.parse_args()
    return args

def compute_mse(args):
    logger.info("Check all the files required.")

    if (not os.path.exists(args.pca_trans_feat_dir)):
        raise Exception("--feat_dir should be existed.")
    else:
        feats_scp = "{feat_dir}/feats.scp".format(feat_dir=args.pca_trans_feat_dir)

    logger.info("Constucting the feature pipes.")
    pca_feats = "ark,s,cs:copy-feats scp:{0} ark:- |".format(feats_scp)

    logger.info("Derive the output using the trained autoencoder")
    if args.cmvn:
        pca_feats = pca_feats + " apply-cmvn --norm-means=true --norm-vars=true --utt2spk=ark:{feat_dir}/utt2spk scp:{feat_dir}/cmvn.scp ark:- ark:- |".format(feat_dir=args.pca_trans_feat_dir)
        print(pca_feats)

    output_dir = '{0}/output'.format(args.dir)

    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    common_lib.execute_command("nnet3-compute --use-gpu=no {ae_nnet} \"{pca_feats}\" ark,scp:{output_dir}/output.ark,{output_dir}/output.scp".format(ae_nnet=args.autoencoder_nnet, pca_feats=pca_feats, output_dir=output_dir))
    common_lib.execute_command("copy-feats \"{pca_feats}\" ark,scp:{output_dir}/input.ark,{output_dir}/input.scp".format(pca_feats=pca_feats, output_dir=output_dir))

    logger.info("Compute MSE.")
    feats_in = output_dir + "/input.scp"
    feats_out = output_dir + "/output.scp"

    autoencdoer_feats_in = kaldi_io.read_mat_scp(feats_in)
    autoencoder_feats_out = kaldi_io.read_mat_scp(feats_out)

    mse_dict = {}
    for d, (utt_in, value_in) in enumerate(autoencdoer_feats_in):
        (utt_out, value_out) = autoencoder_feats_out.next()
        if utt_in != utt_out:
            sys.exit("Mismatch of the input and output of the autoecoder")

        mse_tmp = value_in - value_out
        mse = np.sum(mse_tmp * mse_tmp, axis=1)
        mse = np.mean(mse)

        mse_dict[d] = mse

    logger.info("Store the mse dictionary as a pickle object in local disk.")
    pickle.dump(mse_dict, open("{0}/utt_mse.mse".format(args.dir), "wb"))


def main():
    # process args
    args = get_args()
    try:
        compute_mse(args)
        common_lib.wait_for_background_commands()
    except BaseException as e:
        # look for BaseException so we catch KeyboardInterrupt, which is
        # what we get when a background thread dies.
        if args.email is not None:
            message = ("Computing session for experiment {dir} "
                       "died due to an error.".format(dir=args.dir))
            common_lib.send_mail(message, message, args.email)
        if not isinstance(e, KeyboardInterrupt):
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
