import sys
import numpy as np
from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import QTimer

from audio_manager import AudioManager
from gui import GUI
from utils.custom_enum import FPS, SampleWindowSize

### CUSTOM VARIABLES ###

# Set Visualizer FPS
FRAMERATE = FPS.FPS_30
# Set sample window size to retrieve real time from the audio
SAMPLE_WINDOW_SIZE = SampleWindowSize.WS_1024

########################

class VisualizerApp:
    def __init__(self):
        # Inizializza audio e gui managers
        self.audio_system = AudioManager()
        self.window = GUI(self.audio_system)
        
        # Configura il timer loop
        self.timer = QTimer()
        self.timer.timeout.connect(self.update_loop)
        self.timer.start(FRAMERATE)

        # Avvia la GUI
        self.window.show()

    def update_loop(self):
        # Recupero un chunk audio di una finestra temporale
        chunk = self.audio_system.get_current_chunk(window_size=SAMPLE_WINDOW_SIZE)
        
        # Se il chunk è vuoto o pieno di zeri (silenzio/stop) non fa nulla
        if len(chunk) == 0:
            return

        # test con calcolo volume real time 
        volume = np.linalg.norm(chunk)

        if volume > 0.01:
            print(f"Visual Loop Running... Vol: {volume:.4f}", end='\r')

        # TODO: Invia i dati a TouchDesigner



def main():
    app = QApplication(sys.argv)
    controller = VisualizerApp()
    sys.exit(app.exec())

if __name__ == "__main__":
    main()