## Progress and Ideation

There are many features that can be extracted from an audio file. since the project requires to be able to deal with only harmonic part of the audio, I am thinking of using `hpss` (harmonic-percussive source separation).
![alt text](statics/full%20power%20spectrogram.png)
All percussive spectrogram lines are vertical and harmonic lines are horizontal. So once any one is suppressed, we can get the desired spectrogram.

[Amazing explanation on HPSS here](https://www.youtube.com/watch?v=_AL-SdVem0g)

Other features that can be extracted:

- STFT ( Short Term Fourier Tranform ) and its inverse
- CQT (Constant Q Transform) measures energy in each pitch
  - on display `cqt_hz` shows frequency, `cqt_note` shows note on y axis!
- Chroma - measures energy in each pitch class
- Mel Spectrogram
- MFCC
- Tonnetz
- track beats/bpm of the song

[Librosa Audio Signal Analysis](https://www.youtube.com/watch?v=MhOdbtPhbLU)

**Doubts:**

- [ ] Find a way to club the generated chords
- [ ] Find a way to segment the audio when running inference. (Either `librosa.util.sync` or how?) And how will it work without any percussive instrument to give tempo, thus there might not be a valid frame width to follow.
- [ ] Experiment with deeper and/or residual networks
- [ ] Understand how and why the `n_fft` parameter is affecting the pcp vectors in preprocessing
- [ ] Experiment different activations
- [ ] Read through the CENS paper and go through `libfmp` repo

**Improvements**

- [ ] What I've generated as CSV is actually not CSV. But if i change it there's some more code that needs to change along with it. If time permits, need to refractor the code.

**Progress finished:**

- [x] Make a chord dictionary to map chords. Like ['E,'B'] should give E5. ['C','E','G'] should give Cmajor and so on.
  - Done in chord mapping.ipynb
- [x] What is better for this use case feed forward NN or CNN?

  There's no point in using CNN. I have fixed dimensional data.

- [x] Understanding the working of used synchronisation

  Instead of thinking of audio as one huge file, take it as beat wide frames since a musical chord is beat dependent and not timestamp (second) dependant. Instead of using `librosa.util.sync` we're taking the median of the chroma_cqt arrays that are generated. This guarantees the size of (12,1) even with or without the audio segment being long enough.

- [x] Find and learn python library to deal with ARFF

  Not particularly required. Just the 7th line onwards the whole file can beb thought to be a csv

**Feature Plan:**

- [ ] Add chord transposer option on generation
- [ ] How should output of the look like? Cause we wont have lyrics of the song provided, so chords need to shown according to timestamps when they get changed. Or have an audio playback option and chords keep changing at the timestamps

<hr/>

**Note for self:**

`dataset preprocessor.ipynb` is the main file for preproc. First code block is multithreaded by GPT. Second one is single threaded. (Multiprocessing somehow didnt run)
It needs to be run in my Windows environment.

`model_maker.py/ipynb` is main file for the training and evaluating the model. This is to be run under WSL environment. Probably I'll manage this better making a container later, but for now it works.

`pcp_module.py/ipynb` is made to be used as a module. (Dont delete \_\_init\_\_.py!) (ipynb file was used to try around stuff that worked, py is working)

`pcpvectors` is the directory used to store preproc data as I couldnt have had the patience to wait for a model to be trained just to encounter some error and lose all progress.
