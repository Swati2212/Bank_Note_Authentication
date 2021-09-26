import streamlit as st
import pickle


pickle_in = open('classifier.pkl', 'rb')
classifier = pickle.load(pickle_in)


# @app.route("/")
def welcome():
    return "welcome all"


# @app.route("/predict")
def predict_note_authentication(variance, skewness, curtosis, entropy):
    """Let's Authenticate the Banks Note
            This is using docstrings for specifications.
            ---
            parameters:
              - name: variance
                in: query
                type: number
                required: true
              - name: skewness
                in: query
                type: number
                required: true
              - name: curtosis
                in: query
                type: number
                required: true
              - name: entropy
                in: query
                type: number
                required: true
            responses:
                200:
                    description: The output values

        """
    prediction = classifier.predict([[variance, skewness, curtosis, entropy]])
    print(prediction)
    return prediction

def main():
    st.title("Bank Authenticator")
    html_temp = """
    <div style = "background-color:tomato;padding:10px">
    <h2 style = "color:white;text-align:center;">Streamlit Bank Authenticator ML App</h2>
    </div>
    
    """
    st.markdown(html_temp, unsafe_allow_html=True)
    variance = st.text_input("variance", "Type Here")
    skewness = st.text_input("skewness", "Type Here")
    curtosis = st.text_input("curtosis", "Type Here")
    entropy = st.text_input("entropy", "Type Here")
    result = ""
    if st.button("Predict"):
        result = predict_note_authentication(variance, skewness, curtosis, entropy)
    st.success(f"The Output is {result}")
    if st.button("About"):
        st.text("Lets Learn!")
        st.text("Built With Streamlit!")


if __name__ == "__main__":
    main()
