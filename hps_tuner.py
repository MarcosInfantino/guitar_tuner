import copy
import os
import threading as th
import time

import numpy as np
import scipy.fftpack
import sounddevice as sd

# Settings generales
SAMPLE_FREQ = 48000 # frecuencia de muestreo en Hz
WINDOW_SIZE = 48000 # tamaño de la ventana de la DFT medida en muestras
WINDOW_STEP = 12000 # step size of window
NUM_HPS = 5 # máximo de espectros armónicos a utilizar
POWER_THRESH =2e-5 # límiter de energía de la señal a apartir del cual se activa el afinador
CONCERT_PITCH = 440
WHITE_NOISE_THRESH = 0.2 # todo lo que esté por debajo de WHITE_NOISE_THRESH*avg_energy_per_freq es considerado WHITE NOISE y no se tiene en cuenta

WINDOW_T_LEN = WINDOW_SIZE / SAMPLE_FREQ # longitud de la ventana en segundos
SAMPLE_T_LENGTH = 1 / SAMPLE_FREQ # longitud entre dos muestras en segundos
DELTA_FREQ = SAMPLE_FREQ / WINDOW_SIZE # frequency step width of the interpolated DFT
OCTAVE_BANDS = [50, 100, 200, 400, 800, 1600, 3200, 6400, 12800, 25600]

ALL_NOTES = ["A","A#","B","C","C#","D","D#","E","F","F#","G","G#"]

def find_closest_note(pitch):
  """
  Encuentra la nota más cercana para un pico en la señal
  """
  i = int(np.round(np.log2(pitch/CONCERT_PITCH)*12))
  closest_note = ALL_NOTES[i%12] + str(4 + (i + 9) // 12)
  closest_pitch = CONCERT_PITCH*2**(i/12)
  return closest_note, closest_pitch

HANN_WINDOW = np.hanning(WINDOW_SIZE)
def callback(indata, frames, time, status):
  # define static variables
  if not hasattr(callback, "window_samples"):
    callback.window_samples = [0 for _ in range(WINDOW_SIZE)]#saca el muestro del mic
  if not hasattr(callback, "noteBuffer"):
    callback.noteBuffer = ["1","2"]

  if status:
    print(status)
    return
  if any(indata):
    callback.window_samples = np.concatenate((callback.window_samples, indata[:, 0])) # agrega los nuevos samples
    callback.window_samples = callback.window_samples[len(indata[:, 0]):] # remueve los samples de la afinación anterior

    # si la intensidad de la señal es muy baja, skipea esta iteración
    signal_power = (np.linalg.norm(callback.window_samples, ord=2, axis=0)**2) / len(callback.window_samples)
    if signal_power < POWER_THRESH:
      os.system('cls' if os.name=='nt' else 'clear')
      return

    # multiplicación por ventana de Hann para evitar spectral leakage
    hann_samples = callback.window_samples * HANN_WINDOW
    magnitude_spec = abs(scipy.fftpack.fft(hann_samples)[:len(hann_samples)//2])

    # supresión de ruidos por debajo de los 62hz, con esta supresión aún puede afinarse hasta drop C
    for i in range(int(62/DELTA_FREQ)):
      magnitude_spec[i] = 0

    # se calcula la energía promedio de la señal entera
    # y se suprime toda señal con una energía menor
    # esto permite suprimir el white noise (HPS no logra suprimir el white noise de la mejor manera)
    for j in range(len(OCTAVE_BANDS)-1):
      ind_start = int(OCTAVE_BANDS[j]/DELTA_FREQ)
      ind_end = int(OCTAVE_BANDS[j+1]/DELTA_FREQ)
      ind_end = ind_end if len(magnitude_spec) > ind_end else len(magnitude_spec)
      avg_energy_per_freq = (np.linalg.norm(magnitude_spec[ind_start:ind_end], ord=2, axis=0)**2) / (ind_end-ind_start)
      avg_energy_per_freq = avg_energy_per_freq**0.5
      for i in range(ind_start, ind_end):
        magnitude_spec[i] = magnitude_spec[i] if magnitude_spec[i] > WHITE_NOISE_THRESH*avg_energy_per_freq else 0

    # se interpola el espectro
    # esto es necesario, ya que más abajo sea downsamplea cuando se hace el HPS
    mag_spec_ipol = np.interp(np.arange(0, len(magnitude_spec), 1/NUM_HPS), np.arange(0, len(magnitude_spec)),
                              magnitude_spec)
    mag_spec_ipol = mag_spec_ipol / np.linalg.norm(mag_spec_ipol, ord=2, axis=0)

    hps_spec = copy.deepcopy(mag_spec_ipol)

    # se calcula el HPS para disminuir la intensidad de los armónicos
    for i in range(NUM_HPS):
      tmp_hps_spec = np.multiply(hps_spec[:int(np.ceil(len(mag_spec_ipol)/(i+1)))], mag_spec_ipol[::(i+1)])
      if not any(tmp_hps_spec):
        break
      hps_spec = tmp_hps_spec

    max_ind = np.argmax(hps_spec)
    max_freq = max_ind * (SAMPLE_FREQ/WINDOW_SIZE) / NUM_HPS

    closest_note, closest_pitch = find_closest_note(max_freq)
    max_freq = round(max_freq, 1)
    closest_pitch = round(closest_pitch, 1)

    callback.noteBuffer.insert(0, closest_note) # note that this is a ringbuffer
    callback.noteBuffer.pop()

    os.system('cls' if os.name=='nt' else 'clear')
    if callback.noteBuffer.count(callback.noteBuffer[0]) == len(callback.noteBuffer):
      result = int(max_freq) - int(closest_pitch)
      print("")
      print("---------------------------------------------------------------------------------------------------")
      print( f"Nota más cercana: {closest_note} {max_freq}/{closest_pitch}")
      if result>0:
        print(f"Resultado de la afinación: +{int(max_freq)-int(closest_pitch)}")
      else:
        print(f"Resultado de la afinación: {int(max_freq) - int(closest_pitch)}")
      print("---------------------------------------------------------------------------------------------------")
  else:
    print('no input')


def listen_audio():
  try:
    print("Iniciando el afinador...")
    with sd.InputStream(channels=1, callback=callback, blocksize=WINDOW_STEP, samplerate=SAMPLE_FREQ):
      while True:
        time.sleep(0.5)
  except Exception as exc:
    print(str(exc))

listen_audio()
