This will be chapter about experiments in a two layer fluid using
interface visualisation.

What is going to be in here?

Overview of the methods:

- Photos and camera setup (inc position). 

- post processing - how to deal with parallax and stich images
  together.

Overview of the results:

Images of all of the hovmollers for the runs (the 24fps ones at
least).

Some chat about the current literature, inc. a focus on white and
helfrich and typing of the flows.

A focus on the amplitude of the waves that are produced. Can this be
linked to white and helfrich theory? Energy exchange?

## Intro

The behaviour of gravity currents in a single layer fluid is well
characterised.

The two layer fluid is an important construct in the environment,
see inversions, SAL, etc.

In this chapter we will explore the interactions that a gravity
current has with a two layer fluid.


## Methods

We performed lock release experiments in a fairly standard setup: a
long perspex tank with a removable lock gate at one end.

We pointed a couple of cameras at the tank and recorded 1080p video
at 24fps.

There are three components to the fluid:

- clear upper layer
- green lower layer
- red gravity current

Post processing allows us to measure the position of the interfaces
between these fluids.

```python
from labwaves import RawRun, ProcessedRun

index = 'r13_01_11c'

raw = RawRun(index)
raw.process()

processed = ProcessedRun(index)

x, z, t = processed.combine_wave
```


## Results

Here are some hovmollers showing the evolution of the gravity
current interface (left) and the wave interface (right)...
