from transformers import AutoTokenizer, AutoProcessor, AutoModelForCTC
import torch
import numpy as np
import pandas as pd
from tqdm.auto import tqdm
from pathlib import Path
import argparse

import librosa
from moviepy import editor

parser = argparse.ArgumentParser(description='Use a Deep Learning Model to Detect and Remove Pauses from Your Video')
parser.add_argument('--video_path', required=True, type=str, help='Path to your Raw Video')
parser.add_argument('--output_filename', default='./test.mp4', type=str, help='Path to the output file to be created')
parser.add_argument('--pad', default=0.3, type=float, help='Amount of paddding (seconds) to be added to each timestamp for smoother output')


parser.add_argument('--cache_dir', default='./', type=str, help='(Optional) Path for Model Files to be Saved')
parser.add_argument('--model_id', default="patrickvonplaten/wav2vec2-base-100h-with-lm", type=str, help='(Optional) Name of the Pre-Trained Model on HuggingFace.co')
parser.add_argument('--keep_range', default=None, help='(Optional) Add timestamps for the model to ignore')
parser.add_argument('--timestamp_file', default='./time_stamps.txt', type=str, help='(Optional) Path to save the time stamps created by the model')
parser.add_argument('--resume_from_timestamps', default=False, type=bool, help='Whether to resume editing the video from the saved time stamps')



def main():
    args = parser.parse_args()
    
    model_id = args.model_id

    model = AutoModelForCTC.from_pretrained(model_id, cache_dir=args.cache_dir)
    processor = AutoProcessor.from_pretrained(model_id, cache_dir=args.cache_dir)

    video = editor.VideoFileClip(args.video_path)
    audio = video.audio.to_soundarray() # Outputs audio with sample_rate==44_100 by default
    audio = audio[:, 0] # Take only the first channel, because the model accepts mono audio

    sample_rate=16_000 # Pretrained Model sample rate

    audio = librosa.resample(audio, orig_sr=44_100, target_sr=sample_rate)

    '''
    If the video is too long, you can choose to take chunks of it, 
    process the chunks with the model, and go to the next chunk

    take 60-second chunks:
    chunksize = 60 # seconds
    '''
    chunksize = 60 # seconds
    intervals=[]
    for p in tqdm(range(1,int(audio.shape[0]/16_000/chunksize)+2)):
    #     if p==3: break
        chunk = audio[(p-1)*sample_rate*chunksize:p*sample_rate*chunksize]
        p_start = (p-1)*chunksize
        p_end = p*chunksize

        chunk = processor(chunk, return_tensors='pt', sampling_rate=16_000).input_values
        chunk.shape

        model.eval()
        with torch.no_grad():
            logits = model(chunk).logits

        output = processor.decode(logits[0].numpy(), output_word_offsets=True)
        pad = args.pad
        time_stamps = [
            [(offset['start_offset']*0.02)-pad+p_start, (offset['end_offset']*0.02)+pad+p_start]
        for offset in output['word_offsets']
        ]
        intervals.extend((time_stamps))
    
    interval_union = get_union(intervals)

    # optionally add timestamps to not be trimmed by the model
    if args.keep_range:
        keep_range = [
            [2*60+50., 3*60.],
            [55*60+43., 55*60+53.]
                ]
        interval_union.extend(keep_range)
        interval_union = get_union(interval_union)
        for skeep, ekeep in keep_range:
            for interval in interval_union:
                if interval[0]>=skeep and interval[1]<=ekeep:
                    interval_union.remove(interval)
                if interval[0]>=skeep and interval[0]<=ekeep:
                    interval[0] = skeep
                if interval[1]>=skeep and interval[1]<=ekeep:
                    interval[1] = ekeep
        interval_union = sorted(interval_union, key=lambda x: x[0])
    
    # save the time_stamps extracted by the model
    with open(args.timestamp_file, 'w') as f:
        for line in interval_union:
            f.write(str(line[0]))
            f.write(',')
            f.write(str(line[1]))
            f.write('\n')

    if args.resume_from_timestamps:
        df = pd.read_csv(args.timestamp_file, header=None, names=['start', 'end'])
        interval_union = list(zip(df.start, df.end))

    # create new video clip from time stamps
    newVideo=[]
    for ts in tqdm(interval_union):
        start = ts[0]
        end = ts[1]
        if start<0: start=0
        if end>video.duration: end=video.duration
        newVideo.append(video.subclip(start, end))

    final_video = editor.concatenate_videoclips(newVideo)
    final_video.write_videofile(args.output_filename,
                            audio_fps=48000, audio_codec='aac',
                            codec='libx264',
                            # threads=4, # set if you want to use multiple threads on your CPU for faster encoding 
                            )
def get_union(intervals):
    '''
    Goes through the list 

    args:
    interval: list of intervals in the form of [start_time, end_time]
    '''
    intervals_srt = sorted(intervals, key=lambda x: x[0])

    interval_union=[]
    for i, l in enumerate(intervals_srt):
        if i==0:
            interval_union.append(l)
            continue
            
        prev = interval_union.pop()
        if l[0]<prev[1]:
            interval_union.append([prev[0], l[1]])
        else:
            interval_union.extend([prev, l])

    
    return sorted(interval_union, key=lambda x: x[0])

if __name__=='__main__':
    main()






