from flask import Flask
from sentence_transformers import SentenceTransformer, util
app = Flask(__name__)

@app.route('/')
def hello_world():
    return 'Hello, World!'

# Define a route that accepts a name as a parameter
@app.route('/greet/<name>')
def greet(name):
    # Define the Sentence Transformers model
    model_name = 'bert-base-nli-mean-tokens'  # You can choose a different model if needed
    model = SentenceTransformer(model_name)

    centroids = ["male female man woman boy girl transgender non-binary genderqueer genderfluid androgynous cisgender intersex gender-neutral masculine feminine identity expression sexuality queer",
             "white black asian hispanic native american middle eastern indigenous multiracial pacific islander caucasian african european latino biracial afro-latinx arab south asian east asian caribbean afro-caribbean"
              "liberal conservative progressive socialist capitalist libertarian communist anarchist feminist environmentalist centrist nationalist populist authoritarian radical moderate extremist neoconservative social democrat"]
    
    centroid_embeddings = []
    for centroid in centroids:
        centroid_embeddings.append(model.encode(centroid, convert_to_tensor=True))
    
    line_embeddings = model.encode(name, convert_to_tensor=True)
    similarity = []
    for centroid_embedding in centroid_embeddings:
        
         similarity.append(util.pytorch_cos_sim(line_embeddings, centroid_embedding)[0][0].item())
    maxima = max(similarity)
    maxima_index = similarity.index(maxima)
    return f'Hello, {maxima}!'

if __name__ == '__main__':
    app.run(debug=True)
