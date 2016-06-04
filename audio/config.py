__author__ = 'anushabala'


class DataConfig:
    def __init__(self, mp3_dir='../data/audio/train', wav_dir='../data/audio/train_wav',
                 mappings_file='../lastfm_train_mappings.txt', reload_data=True, save_data=True,
                 pickle_file='../data/audio/train_data.pkl', max_files=-1, max_examples=-1,
                 stats_file='../eval_stats.json', train_split=0.8, params_file='params/audio-model', save_every=1):
        self.reload = reload_data
        self.pickle_file = pickle_file
        self.mp3_dir = mp3_dir
        self.wav_dir = wav_dir
        self.max_files = max_files
        self.max_examples = max_examples
        self.mappings_file = mappings_file
        self.save_data = save_data
        self.stats_file = stats_file
        self.train_split = train_split
        self.params_file = params_file
        self.save_every = save_every

    def to_json(self):
        return {k:v for (k,v) in self.__dict__.items()}


class ModelConfig:
    def __init__(self, model_type='lstm', learning_rate=0.001, reg=0, hidden_size=100, num_epochs=5, verbose=False,
                 batch_size=5, sampling_rate=22050, time_block=None, block_size=11025, use_fft=True, max_audio_len=15, num_labels=2,
                 print_every=5, evaluate_every=1, anneal_every=1, anneal_by=0.95, filter_height=11, filter_width=1,
                 combine_type='concat', normalize=False):
        self.sampling_rate = sampling_rate
        if time_block:
            self.block_size = time_block * sampling_rate
        else:
            self.block_size = block_size

        self.use_fft = use_fft
        self.max_audio_len = max_audio_len # seconds
        self.max_seq_len = (sampling_rate * max_audio_len)/self.block_size
        self.model_type = model_type
        self.lr = learning_rate
        self.reg = reg
        self.hidden_size = hidden_size
        self.batch_size = batch_size
        self.num_labels = num_labels
        self.num_epochs = num_epochs
        self.input_dimension = self.block_size
        # self.input_dimension = self.block_size * 2 if use_fft else self.block_size
        self.verbose = verbose
        self.print_every = print_every
        self.evaluate_every = evaluate_every
        self.anneal_every = anneal_every
        self.anneal_by = anneal_by
        self.filter_heights = filter_height
        self.filter_widths = filter_width
        self.combine_type = combine_type
        self.normalize = normalize

    def to_json(self):
        return {k:v for (k,v) in self.__dict__.items()}