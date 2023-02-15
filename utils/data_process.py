import torch 
import torchaudio 
import random 

class AudioProcessor :
    
    def __init__(
        self,
        sample_rate : int = 22050,
        clip_len : int = 2,
        pad_threshold : float = 0.3,
        n_channel : int = 2,
        n_mels : int = 128,
        n_fft : int = 1024,
        hop_len : int = None,
        top_db : int = 80,
        salience : int = 1
        ) :
        
        self.sr = sample_rate
        self.cl = clip_len
        self.pad_threshold = pad_threshold
        self.n_channel = n_channel
        self.n_mels = n_mels 
        self.n_fft = n_fft 
        self.hl = hop_len 
        self.top_db = top_db 
        self.salience = salience 
        
        self.__MelSpectrogram_processor = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_len,
            n_mels=n_mels
            )
        
        self.__AmplitudeToDB_processor = torchaudio.transforms.AmplitudeToDB(
            top_db=top_db
        )
        
        self.__resampler = lambda sig, curr_sr : torchaudio.transforms.Resample(curr_sr, sample_rate)(sig)
    
    def __open_audio_file(self, file_path : str) :
        sig, sr = torchaudio.load(file_path, normalize=True)
        return (sig, sr)
    
    def __rechannel(self, sig : torch.TensorType) :
        curr_n_channel, n_frame = sig.shape 
        
        if curr_n_channel > self.n_channel :
            raise Exception(f"Channel downsampling from {curr_n_channel}-channel to {self.n_channel}-channel is not supported")
        if self.n_channel % curr_n_channel != 0 :
            raise Exception(f"Channel upsampling from {curr_n_channel}-channel to {self.n_channel}-channel is not valid")
        return sig.repeat(self.n_channel // curr_n_channel, 1)
    
    def __cut_clip(self, sig : torch.TensorType) :
        n_clip_frames = self.sr * self.cl
        n_channel, n_curr_frames = sig.shape
        
        res_l = n_curr_frames % n_clip_frames
        to_pad_l = n_clip_frames - res_l
        if to_pad_l / n_clip_frames >= self.pad_threshold or  n_curr_frames < n_clip_frames:
            pad_begin_l = random.randint(0,to_pad_l)
            pad_end_l = to_pad_l - pad_begin_l
            
            zero_pad_begin = torch.zeros((n_channel, pad_begin_l))
            zero_pad_end = torch.zeros((n_channel, pad_end_l))
            
            sig = torch.cat((zero_pad_begin, sig, zero_pad_end), dim = 1)
        
        else :
            sig = sig[:,:n_curr_frames - res_l]
            
        sig_batch = torch.split(sig.unsqueeze(0), n_clip_frames, dim = -1)
        
        return torch.cat(sig_batch, dim = 0)
    
    def __process(self, sig :  torch.TensorType, sr : int) :
        sig = self.__resampler(sig, sr)
        sig = self.__rechannel(sig)
        #sig_batch : (n_clips, n_channel, n_frame)
        sig_batch = self.__cut_clip(sig)
        melspec_batch = self.__MelSpectrogram_processor(sig_batch)
        melspec_db_batch = self.__AmplitudeToDB_processor(melspec_batch)
        
        return melspec_db_batch
    
    def process(self, file_path : str) :
        sig, sr = self.__open_audio_file(file_path)
        return self.__process(sig, sr)
        


    