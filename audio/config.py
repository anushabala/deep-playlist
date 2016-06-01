__author__ = 'anushabala'


class DataConfig:
    def __init__(self, mp3_dir='../data/audio/train', wav_dir='../data/audio/train_wav',
                 mappings_file='../lastfm_train_mappings.txt', reload_data=True, save_data=True,
                 pickle_file='../data/audio/train_data.pkl', max_files=-1, max_examples=-1):
        self.reload = reload_data
        self.pickle_file = pickle_file
        self.mp3_dir = mp3_dir
        self.wav_dir = wav_dir
        self.max_files = max_files
        self.max_examples = max_examples
        self.mappings_file = mappings_file
        self.save_data = save_data


class ModelConfig:
    def __init__(self, model_type='lstm', learning_rate=0.001, reg=0, hidden_size=100, num_epochs=5, verbose=False,
                 batch_size=5, sampling_rate=22050, block_size=11025, use_fft=True, max_audio_len=15, num_labels=2,
                 print_every=5):
        self.sampling_rate = sampling_rate
        self.block_size = block_size
        self.use_fft = use_fft
        self.max_audio_len = max_audio_len # seconds
        self.max_seq_len = (sampling_rate * max_audio_len)/block_size
        self.model_type = model_type
        self.lr = learning_rate
        self.reg = reg
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.input_dimension = self.block_size * 2 if use_fft else self.block_size
        self.verbose = verbose
        self.print_every = print_every
