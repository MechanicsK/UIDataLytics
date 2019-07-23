from tkinter import *
import time
root =Tk()
 
canvas=Canvas(root,bg="blue")
photo=PhotoImage( file="C://Users//Shivamani//Desktop//UILogo.png")
canvas.grid(row=0,column=1 )
canvas.create_image(0,0,image=photo,anchor="nw")
textv='.'
lbl = Label(root,text=  textv,font=("arial",15),fg="green")
lbl.grid(row=2,column=1)
def updatlevel():
    lbl['text']="Starting{0}".format( countdown(5))
    root.after(1000,updatlevel)
def countdown(count):
    global textv
    while True:
        if textv=='.':
            textv=textv+'.'
        elif textv=='..':
            textv=textv+'.'
        elif textv=='...':
            textv=textv+'.'
        elif textv=='....':
            textv=textv+'.'
        elif textv=='....':
            textv=textv+'.'
        elif textv=='.....':
            textv='..'
        elif textv=='..':
            textv='...'
        elif textv=='...':
            textv='....'
        elif textv=='....':
            textv='.....'
        return textv
        
     
updatlevel()
root.mainloop()
    
