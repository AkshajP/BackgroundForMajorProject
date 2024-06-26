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

using the `get_unique_strong_pitches_for_second` function I can get the strongest pitches heard in a second. Next step is to map it to a chord name.

**Doubts:**

- [ ] get unique strong pitches can produce horrible results when a chord changes in the middle of a second. Giving weird notes that dont add up to a chord. So can I split the audio according to beat-long audiolets instead of second-long audiolets. And would that affect the training of the nueral network later?
- [ ] what does Chroma graph's one column represent?? Is it 1second/length of song? or is it according to sample rate or some other feature. For ex: If it is 1 second, then I can iterate through data of all columns to get strongest notes heard and map it to some chord
- [ ] saw mir_effects multiple places. Need to know what that is
- [ ] Find and learn python library to deal with ARFF and club chords

**Progress finished:**

- [x] Make a chord dictionary to map chords. Like ['E,'B'] should give E5. ['C','E','G'] should give Cmajor and so on.
  - Done in chord mapping.ipynb
