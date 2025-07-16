import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np
import os
import sys
import json
import pickle  # Add this import for loading the pickle file

# Set page configuration
st.set_page_config(
    page_title="Image Caption Generator",
    page_icon="🖼️",
    layout="centered"
)

# Set paths
MODEL_PATH = os.path.join("models", "caption_model.keras")
TOKENIZER_PATH = os.path.join("models", "tokenizer.pkl")  # Add path to tokenizer.pkl

@st.cache_resource
def load_model():
    """Load the caption generation model"""
    model = tf.keras.models.load_model(MODEL_PATH)
    return model

@st.cache_resource
def load_feature_extractor():
    """Load the CNN model for feature extraction"""
    # Using InceptionV3 pretrained on ImageNet, without the top classification layer
    base_model = tf.keras.applications.InceptionV3(weights='imagenet', include_top=False, pooling='avg')
    return base_model

def extract_tokenizer_from_model(model_path):
    """Extract tokenizer from model metadata"""
    # The .keras file is a directory or zip file containing metadata.json
    try:
        # Try to extract the .keras file if it's not already extracted
        import zipfile
        import tempfile
        
        if os.path.isfile(model_path) and not os.path.isdir(model_path):
            # Create a temporary directory
            temp_dir = tempfile.mkdtemp()
            
            # Extract the .keras file (it's a zip file)
            with zipfile.ZipFile(model_path, 'r') as zip_ref:
                zip_ref.extractall(temp_dir)
            
            # Look for metadata.json in the extracted files
            metadata_path = os.path.join(temp_dir, 'metadata.json')
            
            st.write(f"Extracted .keras file to: {temp_dir}")
            st.write(f"Looking for metadata.json at: {metadata_path}")
            
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    st.write(f"Metadata keys: {list(metadata.keys())}")
                    
                    # Check for tokenizer in metadata
                    if 'tokenizer' in metadata:
                        return tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(metadata['tokenizer']))
                    # Sometimes tokenizer might be stored with a different key
                    elif 'word_index' in metadata:
                        # Create a tokenizer from the word_index
                        tokenizer = tf.keras.preprocessing.text.Tokenizer()
                        tokenizer.word_index = metadata['word_index']
                        tokenizer.index_word = {v: k for k, v in metadata['word_index'].items()}
                        return tokenizer
        
        # If the model_path is already a directory, look for metadata.json
        elif os.path.isdir(model_path):
            metadata_path = os.path.join(model_path, 'metadata.json')
            if os.path.exists(metadata_path):
                with open(metadata_path, 'r') as f:
                    metadata = json.load(f)
                    st.write(f"Metadata keys: {list(metadata.keys())}")
                    
                    if 'tokenizer' in metadata:
                        return tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(metadata['tokenizer']))
                    elif 'word_index' in metadata:
                        tokenizer = tf.keras.preprocessing.text.Tokenizer()
                        tokenizer.word_index = metadata['word_index']
                        tokenizer.index_word = {v: k for k, v in metadata['word_index'].items()}
                        return tokenizer
        
        # Also check the metadata.json in the parent directory of the model
        metadata_path = os.path.join(os.path.dirname(model_path), 'metadata.json')
        if os.path.exists(metadata_path):
            with open(metadata_path, 'r') as f:
                metadata = json.load(f)
                st.write(f"Metadata keys from parent dir: {list(metadata.keys())}")
                
                if 'tokenizer' in metadata:
                    return tf.keras.preprocessing.text.tokenizer_from_json(json.dumps(metadata['tokenizer']))
                elif 'word_index' in metadata:
                    tokenizer = tf.keras.preprocessing.text.Tokenizer()
                    tokenizer.word_index = metadata['word_index']
                    tokenizer.index_word = {v: k for k, v in metadata['word_index'].items()}
                    return tokenizer
                    
    except Exception as e:
        st.error(f"Error extracting tokenizer from model: {str(e)}")
    
    # If all attempts fail, return None
    return None

@st.cache_resource
def get_tokenizer():
    """Load tokenizer from pickle file"""
    try:
        # Try to load the tokenizer from the pickle file
        with open(TOKENIZER_PATH, 'rb') as f:
            tokenizer = pickle.load(f)
            st.success("Tokenizer loaded successfully from pickle file!")
            return tokenizer
    except Exception as e:
        st.error(f"Error loading tokenizer from pickle file: {str(e)}")
        
        # Fall back to extracting from model metadata
        tokenizer = extract_tokenizer_from_model(MODEL_PATH)
        if tokenizer is not None:
            return tokenizer
            
        # Create a basic tokenizer if all else fails
        st.warning("Using a basic tokenizer as fallback.")
        tokenizer = tf.keras.preprocessing.text.Tokenizer()
        # Add special tokens
        tokenizer.word_index = {'startseq': 1, 'endseq': 2}
        return tokenizer

def preprocess_image(image, target_size=(299, 299)):
    """Preprocess the image for feature extraction"""
    if image.mode != "RGB":
        image = image.convert("RGB")
    image = image.resize(target_size)
    image = np.array(image)
    # Normalize the image using InceptionV3 preprocessing
    image = image / 127.5 - 1.0
    image = np.expand_dims(image, axis=0)
    return image

def extract_features(image):
    """Extract features from image using InceptionV3"""
    # Load the feature extractor
    feature_extractor = load_feature_extractor()
    
    # Extract features (2048-dimensional vector)
    features = feature_extractor.predict(image, verbose=0)
    
    return features

def inspect_model(model):
    """Inspect model architecture and print details"""
    details = {
        "Inputs": [],
        "Outputs": [],
        "Layers": []
    }
    
    # Get input details
    for i, input_layer in enumerate(model.inputs):
        details["Inputs"].append({
            "Name": input_layer.name,
            "Shape": str(input_layer.shape)
        })
    
    # Get output details
    for i, output_layer in enumerate(model.outputs):
        details["Outputs"].append({
            "Name": output_layer.name,
            "Shape": str(output_layer.shape)
        })
    
    # Get layer details
    for layer in model.layers:
        layer_info = {
            "Name": layer.name,
            "Type": layer.__class__.__name__
        }
        
        # Safely get output shape
        try:
            if hasattr(layer, 'output_shape'):
                layer_info["Shape"] = str(layer.output_shape)
            elif hasattr(layer, 'output'):
                layer_info["Shape"] = str(layer.output.shape)
            else:
                layer_info["Shape"] = "Unknown"
        except:
            layer_info["Shape"] = "Unknown"
            
        details["Layers"].append(layer_info)
    
    return details

def generate_caption(model, image_features):
    """Generate a caption for the image"""
    # Load the tokenizer
    tokenizer = get_tokenizer()
    
    # Define parameters
    max_length = 34  # Maximum sequence length (from model input shape)
    
    # Check what tokens your tokenizer actually has
    has_start_seq = 'startseq' in tokenizer.word_index
    has_start = 'start' in tokenizer.word_index
    has_end_seq = 'endseq' in tokenizer.word_index
    has_end = 'end' in tokenizer.word_index
    
    # Determine the actual start and end tokens used in your model
    start_word = 'start' if has_start else 'startseq'
    end_word = 'end' if has_end else 'endseq'
    
    # Use the more notebook-like approach for caption generation
    in_text = start_word
    caption = []
    
    # Generate the caption word by word
    try:
        # Generate until we reach end token or max length
        for i in range(max_length):
            # Tokenize the current text
            sequence = tokenizer.texts_to_sequences([in_text])[0]
            
            # Pad the sequence to required length
            sequence = tf.keras.preprocessing.sequence.pad_sequences(
                [sequence], maxlen=max_length
            )
            
            # Predict next word
            predictions = model.predict([image_features, sequence], verbose=0)
            
            # Get the index of the predicted word
            predicted_index = np.argmax(predictions[0])
            
            # Convert index to word
            predicted_word = tokenizer.index_word.get(predicted_index, '')
            
            # If end token or max length reached, stop
            if predicted_word == end_word or i >= max_length-1:
                break
                
            # Check for repetition - if the last 3 words are the same as the current word, skip it
            if len(caption) >= 3 and caption[-1] == predicted_word and caption[-2] == predicted_word and caption[-3] == predicted_word:
                # Try the second most likely word instead
                second_best_index = np.argsort(predictions[0])[-2]
                predicted_word = tokenizer.index_word.get(second_best_index, '')
                
                # If still repeating, just break
                if len(caption) >= 3 and caption[-1] == predicted_word and caption[-2] == predicted_word:
                    break
            
            # Add the predicted word to the caption and input text
            if predicted_word and predicted_word != start_word:
                caption.append(predicted_word)
                in_text += ' ' + predicted_word
    
    except Exception as e:
        st.error(f"Error in caption generation: {str(e)}")
        return f"Error generating caption: {str(e)}"
    
    # Join the words to create the caption
    return ' '.join(caption)

def main():
    st.title("Image Caption Generator")
    st.write("Upload an image and get an AI-generated caption")
    
    # Load model with proper error handling
    with st.spinner("Loading model..."):
        try:
            model = load_model()
            st.success("Model loaded successfully!")
        except Exception as e:
            st.error(f"Error loading model: {str(e)}")
            st.stop()
    
    # Safely display model architecture information
    if st.checkbox("Show model details"):
        try:
            model_details = inspect_model(model)
            st.json(model_details)
        except Exception as e:
            st.error(f"Error inspecting model: {str(e)}")
    
    # Image upload
    uploaded_file = st.file_uploader("Choose an image...", type=["jpg", "jpeg", "png"])
    
    if uploaded_file is not None:
        # Display the image
        image = Image.open(uploaded_file)
        st.image(image, caption="Uploaded Image", use_container_width=True)
        
        # Process image and generate caption
        if st.button("Generate Caption"):
            with st.spinner("Generating caption..."):
                try:
                    # Preprocess the image
                    processed_image = preprocess_image(image)
                    
                    # Extract features from the image
                    image_features = extract_features(processed_image)
                    
                    # Generate caption
                    caption = generate_caption(model, image_features)
                    
                    # Display the caption
                    st.success("Caption Generated!")
                    st.write(f"### Generated Caption:")
                    st.write(f"**{caption}**")
                except Exception as e:
                    st.error(f"Error generating caption: {str(e)}")
                    st.write("Error details:", str(e))

if __name__ == "__main__":
    main()