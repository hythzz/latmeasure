# Audio DAC latency measurement

## Purpose
simple script used to measure latency differences between external audio DAC (e.g. USB DAC) and DAC on motherboard. For gaming, it's helpful to adjust the video output offset in terms of the DAC latency

## Requirement and Installation
- An aux-to-aux cable
- A computer with both output jack and mic-in input jack
- A Windows or Linux OS with python3 installed (tested on Windows 10 and Ubuntu 19.10)
- Run `pip install -r requirement.txt` to install required python dependencies

## Usage
1. Connect aux cable so that audio output directly go to mic input
2. Select default audio output to the port you connected, so as to default input
3. run `python LatencyMeasure.py`
4. program runs for seconds, you will get a latency number. this will be your onboard audio RTT
5. Repeat 1-4 with your DAC connected in the loop, you will get a different RRT number
6. The difference between these two numbers is your DAC latency


## How it works
It's simple, the script reads a sample audio file with a triangle sound lasting for a second. Then records what's received from microphone input. Sliding window and signal diff are used to find the offset of original signal. The offset is then converted to human readable millisecond.

For robustness, the script first checks the freqency response and power of the signal to ensure received signal has a similar freqency response as output signal.