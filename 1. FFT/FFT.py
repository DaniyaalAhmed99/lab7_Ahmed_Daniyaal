
import numpy as np
import matplotlib.pyplot as plt
from scipy import fftpack

np.random.seed(1348)

time_step = 0.02

period = 5.0

time_vec = np.arange(0, 20, time_step)

signal1 = (np.sin(5 * 2*np.pi/period*time_vec) + 
          0.5*np.random.randn(time_vec.size))

signal2 = (np.sin(3 * 3*np.pi/period*time_vec) + 
          0.5*np.random.randn(time_vec.size))

signal3 = (np.sin(2 * 4*np.pi/period*time_vec) + 
          0.5*np.random.randn(time_vec.size))


sig_fft1 = fftpack.fft(signal1)
sig_fft2 = fftpack.fft(signal2)
sig_fft3 = fftpack.fft(signal3)

power1 = np.abs(sig_fft1)**2
power2 = np.abs(sig_fft2)**2
power3 = np.abs(sig_fft3)**2

sample_freq1 = fftpack.fftfreq(signal1.size, d=time_step)
sample_freq2 = fftpack.fftfreq(signal2.size, d=time_step)
sample_freq3 = fftpack.fftfreq(signal3.size, d=time_step)

plt.figure(figsize=(6,5))

plt.plot(sample_freq1, power1)
plt.plot(sample_freq2, power2)
plt.plot(sample_freq3, power3)

plt.xlim(-1.5, 1.5)
plt.xlabel('Frequency [Hz]')
plt.ylabel('Amplitude')

'''
We want to remove all high frequencies!'''
pos_mask = np.where(sample_freq1 > 0)
freqs = (sample_freq1[pos_mask] + sample_freq2[pos_mask] + sample_freq3[pos_mask])/3
peak_freq = freqs[power1[pos_mask].argmax() + power2[pos_mask].argmax() + power3[pos_mask].argmax()]

np.allclose(peak_freq, 1./period)

high_freq_fft1 = sig_fft1.copy()
high_freq_fft1[np.abs(sample_freq1) > peak_freq] = 0
filtered_sig1 = fftpack.ifft(high_freq_fft1)

high_freq_fft2 = sig_fft2.copy()
high_freq_fft2[np.abs(sample_freq2) > peak_freq] = 0
filtered_sig2 = fftpack.ifft(high_freq_fft2)


high_freq_fft3 = sig_fft3.copy()
high_freq_fft3[np.abs(sample_freq3) > peak_freq] = 0
filtered_sig3 = fftpack.ifft(high_freq_fft3)




plt.figure(figsize=(8, 5))
plt.plot(time_vec, signal1, label='Original Signal1')
plt.plot(time_vec, signal2, label='Original Signal2')
plt.plot(time_vec, signal3, label='Original Signal3')
plt.legend(loc = 'best')

plt.figure(figsize=(8, 5))

plt.plot(time_vec, filtered_sig1, label='filtered signal1')
plt.plot(time_vec, filtered_sig2, label='filtered signal1')
plt.plot(time_vec, filtered_sig3, label='filtered signal1')

plt.legend(loc = 'best')



