"""
Joint beat and downbeat tracking with a dynamic Bayesian network (DBN),
vendored from madmom's `DBNDownBeatTrackingProcessor` [1] so that beat_this
does not need madmom at runtime.

Adapted from madmom (https://github.com/CPJKU/madmom, commit 27f032e):
madmom/features/downbeats.py, madmom/features/beats_hmm.py and
madmom/ml/hmm.pyx. Only the code paths used by beat_this are kept. The
Cython Viterbi decoder is replaced by a NumPy implementation that exploits
the structure of the bar state space: every state except the first state of
each beat has a single predecessor (the previous state, with probability 1),
so each frame is a shift plus a small dense max over tempo transitions.

madmom's source code is distributed under the following license:

Copyright (c) 2012-2014 Department of Computational Perception,
Johannes Kepler University, Linz, Austria and Austrian Research Institute for
Artificial Intelligence (OFAI), Vienna, Austria.
All rights reserved.

Redistribution and use in source and binary forms, with or without
modification, are permitted provided that the following conditions are met:

1. Redistributions of source code must retain the above copyright notice, this
   list of conditions and the following disclaimer.
2. Redistributions in binary form must reproduce the above copyright notice,
   this list of conditions and the following disclaimer in the documentation
   and/or other materials provided with the distribution.

THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS" AND
ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE IMPLIED
WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT OWNER OR CONTRIBUTORS BE LIABLE FOR
ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL DAMAGES
(INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR SERVICES;
LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER CAUSED AND
ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY, OR TORT
(INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE OF THIS
SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

References
----------
.. [1] Sebastian Böck, Florian Krebs and Gerhard Widmer,
       "Joint Beat and Downbeat Tracking with Recurrent Neural Networks"
       Proceedings of the 17th International Society for Music Information
       Retrieval Conference (ISMIR), 2016.
.. [2] Florian Krebs, Sebastian Böck and Gerhard Widmer,
       "An Efficient State Space Model for Joint Tempo and Meter Tracking",
       Proceedings of the 16th International Society for Music Information
       Retrieval Conference (ISMIR), 2015.
"""

import warnings

import numpy as np


def beat_intervals(min_interval, max_interval, num_intervals=None) -> np.ndarray:
    """
    Beat intervals (in frames) modeled by the state space [2]. Uses a linear
    spacing, unless `num_intervals` is smaller than the number of linearly
    spaced intervals, in which case a log spacing with (at least)
    `num_intervals` intervals is used.
    """
    intervals = np.arange(np.round(min_interval), np.round(max_interval) + 1)
    if num_intervals is not None and num_intervals < len(intervals):
        # we must approach the number of intervals iteratively
        num_log_intervals = num_intervals
        intervals = []
        while len(intervals) < num_intervals:
            intervals = np.logspace(
                np.log2(min_interval),
                np.log2(max_interval),
                num_log_intervals,
                base=2,
            )
            # quantize to integer intervals
            intervals = np.unique(np.round(intervals))
            num_log_intervals += 1
    return np.ascontiguousarray(intervals, dtype=int)


def exponential_transition(intervals, transition_lambda) -> np.ndarray:
    """
    Exponential tempo transition probabilities between beat intervals [2],
    as a matrix of shape (from_intervals, to_intervals).
    """
    ratio = intervals.astype(float) / intervals.astype(float)[:, np.newaxis]
    prob = np.exp(-transition_lambda * abs(ratio - 1.0))
    # set values below threshold to 0
    prob[prob <= np.spacing(1)] = 0
    # normalize the emission probabilities
    prob /= np.sum(prob, axis=1)[:, np.newaxis]
    return prob


class BarHMM:
    """
    HMM modeling a bar of `num_beats` beats (bar state space, bar transition
    model and RNN downbeat tracking observation model of madmom).

    The state space consists of `num_beats` identical beats. Each beat
    contains one run of consecutive states per beat interval, the length of
    the run being the interval (in frames). Tempo changes happen only when
    moving from the last state of one beat to the first state of the next.
    """

    def __init__(self, num_beats, intervals, transition_lambda, observation_lambda):
        self.num_beats = int(num_beats)
        # states of a single beat
        beat_positions = np.concatenate(
            [np.linspace(0, 1, i, endpoint=False) for i in intervals]
        )
        beat_first_states = np.cumsum(np.r_[0, intervals[:-1]])
        beat_last_states = np.cumsum(intervals) - 1
        states_per_beat = len(beat_positions)
        # stack them `num_beats` times
        offsets = np.arange(self.num_beats)[:, np.newaxis] * states_per_beat
        self.num_states = self.num_beats * states_per_beat
        self.state_positions = np.concatenate(
            [beat_positions + b for b in range(self.num_beats)]
        )
        # first/last states of each beat, shape (num_beats, num_intervals)
        self.first_states = beat_first_states + offsets
        self.last_states = beat_last_states + offsets
        # log transition probabilities from the last states of the previous
        # beat to the first states of the current beat (same for all beats)
        with np.errstate(divide="ignore"):
            self.log_tempo_transitions = np.log(
                exponential_transition(intervals, transition_lambda)
            )
        # observation model: pointers from states to density columns
        # (0: no beat, 1: beat, 2: downbeat)
        self.observation_lambda = observation_lambda
        border = 1.0 / observation_lambda
        self.om_pointers = np.zeros(self.num_states, dtype=np.uint32)
        self.om_pointers[self.state_positions % 1 < border] = 1
        self.om_pointers[self.state_positions < border] = 2

    def log_densities(self, observations) -> np.ndarray:
        """
        Log densities of the observations of shape (N, 2), with columns for
        beat and downbeat probabilities. Returns an array of shape (N, 3) with
        columns for no-beat, beat and downbeat log densities.
        """
        log_densities = np.empty((len(observations), 3), dtype=float)
        with np.errstate(divide="ignore", invalid="ignore"):
            log_densities[:, 0] = np.log(
                (1.0 - np.sum(observations, axis=1)) / (self.observation_lambda - 1)
            )
            log_densities[:, 1] = np.log(observations[:, 0])
            log_densities[:, 2] = np.log(observations[:, 1])
        # madmom's Cython Viterbi never selects a transition with NaN density,
        # which is equivalent to a density of -inf
        log_densities[np.isnan(log_densities)] = -np.inf
        return log_densities

    def viterbi(self, observations) -> tuple[np.ndarray, float]:
        """
        Determine the best path through the state space with the Viterbi
        algorithm, assuming a uniform initial state distribution.

        Returns
        -------
        path : numpy array
            Best state-space path sequence.
        log_prob : float
            Corresponding log probability.
        """
        om_densities = self.log_densities(observations)[:, self.om_pointers]
        num_frames = len(om_densities)
        first_states = self.first_states
        # the first states of each beat are entered from the last states of
        # the previous beat
        prev_last_states = np.roll(self.last_states, 1, axis=0)
        log_tempo_transitions = self.log_tempo_transitions[np.newaxis]
        # back-tracking pointers are only needed for the first states of each
        # beat, since all other states have a single predecessor; they index
        # into the intervals of `prev_last_states`
        bt_pointers = np.empty(
            (num_frames,) + first_states.shape,
            dtype=np.min_scalar_type(first_states.shape[1] - 1),
        )
        viterbi = np.full(self.num_states, -np.log(self.num_states))
        for frame in range(num_frames):
            # all transitions into states that are not the first state of
            # a beat come from the previous state with probability 1
            current = np.empty_like(viterbi)
            current[1:] = viterbi[:-1]
            # transitions into the first states of each beat, shape
            # (num_beats, from_intervals, to_intervals)
            candidates = (
                viterbi[prev_last_states][:, :, np.newaxis] + log_tempo_transitions
            )
            best = candidates.argmax(axis=1)
            bt_pointers[frame] = best
            current[first_states] = np.take_along_axis(
                candidates, best[:, np.newaxis], axis=1
            )[:, 0]
            # weight with the observation densities
            current += om_densities[frame]
            viterbi = current

        # fetch the final best state
        state = int(viterbi.argmax())
        log_probability = viterbi[state]
        # raise warning if the sequence has -inf probability
        if np.isinf(log_probability):
            warnings.warn(
                "-inf log probability during Viterbi decoding "
                "cannot find a valid path",
                RuntimeWarning,
            )
            return np.empty(0, dtype=np.uint32), log_probability

        # map each first state to its (beat, interval) index, -1 otherwise
        first_state_index = np.full(self.num_states, -1)
        first_state_index[first_states.ravel()] = np.arange(first_states.size)
        prev_last_states = prev_last_states.ravel()
        bt_pointers = bt_pointers.reshape(num_frames, -1)
        num_intervals = first_states.shape[1]
        # track the path backwards, starting with the last frame
        path = np.empty(num_frames, dtype=np.uint32)
        for frame in range(num_frames - 1, -1, -1):
            path[frame] = state
            idx = first_state_index[state]
            if idx < 0:
                state -= 1
            else:
                beat = idx // num_intervals
                state = prev_last_states[beat * num_intervals + bt_pointers[frame, idx]]
        return path, log_probability


def threshold_activations(activations, threshold):
    """
    Threshold activations to include only the main segment exceeding the given
    threshold (i.e. first to last time/index exceeding the threshold).
    Returns the thresholded activations and the index of the first frame.
    """
    first = last = 0
    # use only the activations > threshold
    idx = np.nonzero(activations >= threshold)[0]
    if idx.any():
        first = max(first, np.min(idx))
        last = min(len(activations), np.max(idx) + 1)
    # return thresholded activations segment and first index
    return activations[first:last], first


class DBNDownBeatTrackingProcessor:
    """
    Downbeat tracking with RNNs and a dynamic Bayesian network (DBN)
    approximated by a Hidden Markov Model (HMM) [1].

    Parameters
    ----------
    beats_per_bar : int or list
        Number of beats per bar to be modeled. Can be either a single number
        or a list with bar lengths (in beats).
    min_bpm : float, optional
        Minimum tempo used for beat tracking [bpm].
    max_bpm : float, optional
        Maximum tempo used for beat tracking [bpm].
    num_tempi : int, optional
        Number of tempi to model; if set, limit the number of tempi and use a
        log spacing, otherwise a linear spacing.
    transition_lambda : float, optional
        Lambda for the exponential tempo change distribution (higher values
        prefer a constant tempo from one beat to the next one).
    observation_lambda : int, optional
        Split one (down-)beat period into `observation_lambda` parts, the first
        representing (down-)beat states and the remaining non-beat states.
    threshold : float, optional
        Threshold the RNN (down-)beat activations before Viterbi decoding.
    fps : float
        Frames per second.

    Notes
    -----
    As in madmom's default configuration, the detected beats are corrected,
    i.e. aligned to the highest peak of the (down-)beat activations within
    the beat range of the decoded path.
    """

    def __init__(
        self,
        beats_per_bar,
        min_bpm=55.0,
        max_bpm=215.0,
        num_tempi=60,
        transition_lambda=100,
        observation_lambda=16,
        threshold=0.05,
        fps=None,
    ):
        # convert timing information to construct a beat state space
        min_interval = 60.0 * fps / max_bpm
        max_interval = 60.0 * fps / min_bpm
        intervals = beat_intervals(min_interval, max_interval, num_tempi)
        # model the different bar lengths
        self.hmms = [
            BarHMM(beats, intervals, transition_lambda, observation_lambda)
            for beats in np.array(beats_per_bar, ndmin=1)
        ]
        self.threshold = threshold
        self.fps = fps

    def __call__(self, activations) -> np.ndarray:
        """
        Detect the (down-)beats in the given activation function.

        Parameters
        ----------
        activations : numpy array, shape (num_frames, 2)
            Activation function with probabilities corresponding to beats
            and downbeats given in the first and second column, respectively.

        Returns
        -------
        beats : numpy array, shape (num_beats, 2)
            Detected (down-)beat positions [seconds] and beat numbers.
        """
        # use only the activations > threshold (init offset to be added later)
        first = 0
        if self.threshold:
            activations, first = threshold_activations(activations, self.threshold)
        # return no beats if no activations given / remain after thresholding
        if not activations.any():
            return np.empty((0, 2))
        # decode the activations with each HMM and choose the best one
        # (highest log probability)
        results = [hmm.viterbi(activations) for hmm in self.hmms]
        best = np.argmax([log_prob for _, log_prob in results])
        path, _ = results[best]
        hmm = self.hmms[best]
        # corresponding beats (add 1 for natural counting)
        beat_numbers = hmm.state_positions[path].astype(int) + 1
        # for each detection determine the "beat range", i.e. states where
        # the pointers of the observation model are >= 1
        beat_range = hmm.om_pointers[path] >= 1
        # if there aren't any in the beat range, there are no beats
        if not beat_range.any():
            return np.empty((0, 2))
        # get all change points between True and False (cast to int before)
        idx = np.nonzero(np.diff(beat_range.astype(int)))[0] + 1
        # if the first frame is in the beat range, add a change at frame 0
        if beat_range[0]:
            idx = np.r_[0, idx]
        # if the last frame is in the beat range, append the length of the array
        if beat_range[-1]:
            idx = np.r_[idx, beat_range.size]
        # iterate over all regions and pick the frame with the highest
        # activation value; since np.argmax works on the flattened array of
        # beat and downbeat activations, we need to divide by 2
        beats = np.array(
            [
                np.argmax(activations[left:right]) // 2 + left
                for left, right in idx.reshape((-1, 2))
            ],
            dtype=int,
        )
        # return the beat positions (converted to seconds) and beat numbers
        return np.vstack(((beats + first) / float(self.fps), beat_numbers[beats])).T
