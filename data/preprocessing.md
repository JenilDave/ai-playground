# Preprocessing

Before supplying data to the ML models, consider doing processing on it. This process prepares and transforms our data to make it ready for ML

### Feature Processing
As we have our initial dataset ready we need to make sure all relevant features are part of it. This could be joining data set from another Table or file.

Another important steps is the clean the irrelevant records which is part of cleaning. Below is what needs to be done:
- Remove all records where many features are missing (only if they are less in number)
- Remove all features which is not present in many records

Another process involves Feature Engineering, which is combining multiple features into one to draw out a signal

### Data Cleaning
Clean the Data before pushing to the model. It depends on the type of data, what kind of cleaning needs to be applied. Below are some examples of the type of input and possible data cleaning to be done:
- Image: Resizing, Cropping, Clip
- Text: lower, regex, etc.

### Transformation
- Scale the input within 0-1 range. This helps in getting statistics on our data. This process is called standardization. Additional rescalling can be done to bring all values between range of 0 to 1
- Encoding is another process of transforming data. One of them is converting string labels to one of the numbers. Another popular encoding is One-Hot (one vs all). Even generating embeddings is another method

### Extraction
- Extract Signals from existing features / combine existing features
- Apply Transfer Learning which is using pre-trained models to extract some features and supply them as input to your model