from tkinter import Tk, Button, Label, Frame
from PIL import ImageTk
from image_manager import ImageManager
from json_handler import JSONHandler

# Set up the main window
root = Tk()
root.title("Image Labeling Tool")
root.configure(bg='grey')

# Initialize ImageManager and JSONHandler
image_manager = ImageManager()
json_handler = JSONHandler(image_manager)

# Function to update the counter display
def update_counter_display():
    counter_label.config(text=f"{image_manager.images_left} out of {image_manager.total_images}")

# Function to display the next image
def display_next_image():
    img_tk, std_dev_text = image_manager.display_next_image()
    img_label.config(image=img_tk)
    img_label.image = img_tk
    std_dev_label.config(text=std_dev_text)

# Function to save the label for the current image
def save_labeled_image(label):
    json_handler.save_labeled_image(label)
    image_manager.remove_current_image()
    update_counter_display()
    display_next_image()

# Function to undo the last labeled action
def undo_last_label():
    json_handler.undo_last_label()
    update_counter_display()
    display_next_image()

# GUI setup
counter_label = Label(root, text="", bg='grey', font=("Helvetica", 10))
counter_label.grid(row=0, column=0, sticky="w", padx=10, pady=5)

img_label = Label(root, bg='grey')
img_label.grid(row=1, column=0, padx=10, pady=10)

buttons_frame = Frame(root, bg='grey')
buttons_frame.grid(row=3, column=0, padx=10, pady=10)

button_width = 10
button_height = 2

sell_button = Button(buttons_frame, text="Sell", command=lambda: save_labeled_image("sell"),
                     width=button_width, height=button_height)
sell_button.pack(side="left", padx=5, pady=5)

hold_button = Button(buttons_frame, text="Hold", command=lambda: save_labeled_image("hold"),
                     width=button_width, height=button_height)
hold_button.pack(side="left", padx=5, pady=5)

buy_button = Button(buttons_frame, text="Buy", command=lambda: save_labeled_image("buy"),
                    width=button_width, height=button_height)
buy_button.pack(side="left", padx=5, pady=5)

undo_button = Button(buttons_frame, text="Undo", command=undo_last_label,
                     width=button_width, height=button_height)
undo_button.pack(side="left", padx=5, pady=5)

std_dev_label = Label(root, text="Standard Deviation: N/A", bg='grey', font=("Helvetica", 10))
std_dev_label.grid(row=2, column=0, sticky="w", padx=10, pady=5)

# Key binding to handle arrow key input
def key_pressed(event):
    if event.keysym == 'Left':
        save_labeled_image("sell")
    elif event.keysym == 'Down':
        save_labeled_image("hold")
    elif event.keysym == 'Right':
        save_labeled_image("buy")

root.bind('<Left>', key_pressed)
root.bind('<Down>', key_pressed)
root.bind('<Right>', key_pressed)

display_next_image()
update_counter_display()

# Run the application
root.mainloop()
