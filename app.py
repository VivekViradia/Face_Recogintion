
from flask import Flask, render_template, request
import tensorflow as tf
from tensorflow.keras.models import load_model
from keras.preprocessing import image
import numpy as np

app = Flask(__name__)

dic = {0:'pins_Adriana Lima',
 1:'pins_Alex Lawther',
 2:'pins_Alexandra Daddario',
 3:'pins_Alvaro Morte',
 4:'pins_Amanda Crew',
 5:'pins_Andy Samberg',
 6:'pins_Anne Hathaway',
 7:'pins_Anthony Mackie',
 8:'pins_Avril Lavigne',
 9:'pins_Ben Affleck',
 10:'pins_Bill Gates',
 11:'pins_Bobby Morley',
 12:'pins_Brenton Thwaites',
 13:'pins_Brian J. Smith',
 14:'pins_Brie Larson',
 15:'pins_Chris Evans',
 16:'pins_Chris Hemsworth',
 17:'pins_Chris Pratt',
 18:'pins_Christian Bale',
 19:'pins_Cristiano Ronaldo',
 20:'pins_Danielle Panabaker',
 21:'pins_Dominic Purcell',
 22:'pins_Dwayne Johnson',
 23:'pins_Eliza Taylor',
 24:'pins_Elizabeth Lail',
 25:'pins_Emilia Clarke',
 26:'pins_Emma Stone',
 27:'pins_Emma Watson',
 28:'pins_Gwyneth Paltrow',
 29:'pins_Henry Cavil',
 30:'pins_Hugh Jackman',
 31:'pins_Inbar Lavi',
 32:'pins_Irina Shayk',
 33:'pins_Jake Mcdorman',
 34:'pins_Jason Momoa',
 35:'pins_Jennifer Lawrence',
 36:'pins_Jeremy Renner',
 37:'pins_Jessica Barden',
 38:'pins_Jimmy Fallon',
 39:'pins_Johnny Depp',
 40:'pins_Josh Radnor',
 41:'pins_Katharine Mcphee',
 42:'pins_Katherine Langford',
 43:'pins_Keanu Reeves',
 44:'pins_Krysten Ritter',
 45:'pins_Leonardo DiCaprio',
 46:'pins_Lili Reinhart',
 47:'pins_Lindsey Morgan',
 48:'pins_Lionel Messi',
 49:'pins_Logan Lerman',
 50:'pins_Madelaine Petsch',
 51:'pins_Maisie Williams',
 52:'pins_Maria Pedraza',
 53:'pins_Marie Avgeropoulos',
 54:'pins_Mark Ruffalo',
 55:'pins_Mark Zuckerberg',
 56:'pins_Megan Fox',
 57:'pins_Miley Cyrus',
 58:'pins_Millie Bobby Brown',
 59:'pins_Morena Baccarin',
 60:'pins_Morgan Freeman',
 61:'pins_Nadia Hilker',
 62:'pins_Natalie Dormer',
 63:'pins_Natalie Portman',
 64:'pins_Neil Patrick Harris',
 65:'pins_Pedro Alonso',
 66:'pins_Penn Badgley',
 67:'pins_Rami Malek',
 68:'pins_Rebecca Ferguson',
 69:'pins_Richard Harmon',
 70:'pins_Rihanna',
 71:'pins_Robert De Niro',
 72:'pins_Robert Downey Jr',
 73:'pins_Sarah Wayne Callies',
 74:'pins_Selena Gomez',
 75:'pins_Shakira Isabel Mebarak',
 76:'pins_Sophie Turner',
 77:'pins_Stephen Amell',
 78:'pins_Taylor Swift',
 79:'pins_Tom Cruise',
 80:'pins_Tom Hardy',
 81:'pins_Tom Hiddleston',
 82:'pins_Tom Holland',
 83:'pins_Tuppence Middleton',
 84:'pins_Ursula Corbero',
 85:'pins_Wentworth Miller',
 86:'pins_Zac Efron',
 87:'pins_Zendaya',
 88:'pins_Zoe Saldana',
 89:'pins_alycia dabnem carey',
 90:'pins_amber heard',
 91:'pins_barack obama',
 92:'pins_barbara palvin',
 93:'pins_camila mendes',
 94:'pins_elizabeth olsen',
 95:'pins_ellen page',
 96:'pins_elon musk',
 97:'pins_gal gadot',
 98:'pins_grant gustin',
 99:'pins_jeff bezos',
 100:'pins_kiernen shipka',
 101:'pins_margot robbie',
 102:'pins_melissa fumero',
 103:'pins_scarlett johansson',
 104:'pins_tom ellis'}

model = load_model('model.h5')

model.make_predict_function()

global score

def predict_label(img_path):
	i = tf.keras.preprocessing.image.load_img(img_path, target_size=(224,224))
	i = tf.keras.preprocessing.image.img_to_array(i)/255.0
	i = i.reshape(1, 224,224,3)
	score = model.predict(i)
	return dic[np.argmax(score)][5:].title()


# routes
@app.route("/", methods=['GET', 'POST'])
def main():
	return render_template("index.html")

@app.route("/about")
def about_page():
	return "Hello World"

@app.route("/submit", methods = ['GET', 'POST'])
def get_output():
	if request.method == 'POST':
		img = request.files['my_image']

		img_path = "static/" + img.filename	
		img.save(img_path)

		score = predict_label(img_path)

	return render_template("index.html", prediction = score, img_path = img_path)


if __name__ =='__main__':
	#app.debug = True
	app.run(debug = True)