from easydict import EasyDict as edict

config = edict()

# configurations for LDA.
config.N_TOPICS=12

config.DATA_DIR = './data/ChipSeq'

config.OUTPUT = './output/'

config.TOKENS_FILE= 'chipseq.tokens'

config.DATA_FILE='chipseq.ldac'

config.N_ITER=2500

config.ALPHA=0.1

config.ETA=0.01

config.THIN=10

config.BURN_IN=20

config.TOPICS_LIST = [3, 6, 10, 12, 14, 16, 20, 24, 28, 32]