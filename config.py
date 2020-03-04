from easydict import EasyDict as edict

config = edict()

# configurations for LDA.
config.N_TOPICS=10 	# number of topics

config.DATA_DIR = './data/toy_data' # data directory

config.OUTPUT = './output/' # output directory to save graphs

config.TOKENS_FILE= 'toy.tokens' # tokens file

config.DATA_FILE='toy.ldac' # data file

config.N_ITER=500 # number of iterations

config.ALPHA=0.1 # topic prior

config.ETA=0.01 # words prior

config.THIN=10 # to apply thinning

config.BURN_IN=20 # burn in 

config.TOPIC_DISTR = 'word_topic.pkl' # file to store topic-word distribution

config.TEST_DOC = 'toy.ldac'

# This parameter is used to find the optimal number of topics based on log-likelihood convergence
config.TOPICS_LIST = [3, 6, 10, 12, 14, 16, 20, 24, 28, 32]

# parameters specific to evaluation. 
config.NUM_SAMPLES = 1500
