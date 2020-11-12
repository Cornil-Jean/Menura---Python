import tkinter as tk
from tkinter import *
from tkinter import ttk
from menura import *

class menura( Frame ):
    def __init__( self ):
        tk.Frame.__init__(self)
        self.pack()
        self.master.title("Menura: Bird Tracker")
        self.button1 = Button( self, text = "Record", width = 108, height = 20,
                               command = self.rec)
        self.button1.grid( row = 0, column = 1, columnspan = 2, sticky = W+E+N+S )
    def rec(self):
        sample, fs = recsample()
        plotstft(sample, fs)
        sampleCorrelation()

def main():
    menura().mainloop()
def main():
    menura().mainloop()
if __name__ == '__main__':
    main()
