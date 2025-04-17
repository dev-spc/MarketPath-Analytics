# MarketPath Analytics: Multi-Touch Attribution Modeling for Digital Marketing Optimization

## Project Overview

MarketPath Analytics is a comprehensive data science project that implements and compares six attribution models to understand how different marketing channels contribute to customer conversions. By analyzing customer journey data across multiple touchpoints, this project provides actionable insights for optimizing marketing budget allocation and campaign strategies.

## Business Problem

In modern digital marketing, customers interact with brands through multiple channels before making a purchase decision. Traditional single-touch attribution models (like first or last click) fail to capture the full contribution of each marketing channel. This project addresses this challenge by implementing and comparing multiple attribution approaches to provide a more holistic view of marketing effectiveness.

## Features and Capabilities

- **Multi-Model Analysis**: Implements six distinct attribution models ranging from simple rule-based to advanced machine learning approaches
- **Customer Journey Analysis**: Maps and analyzes complete customer journeys from first touchpoint to conversion
- **Channel Contribution Visualization**: Clear visualization of how different models attribute value to marketing channels
- **Data-Driven Insights**: Uses statistical and machine learning techniques to extract patterns from user behavior
- **Business Applications**: Provides actionable insights for marketing budget allocation and campaign optimization

## Data Description

The dataset contains marketing touchpoint data with the following structure:

- **user_id**: Unique identifier for each customer
- **touchpoint**: Marketing channel interaction (email, social, search, display, video, affiliate)
- **timestamp**: Date of the interaction
- **conversion**: Binary flag indicating conversion (1) or non-conversion (0)

The data reflects realistic customer journeys across marketing channels with approximately 3,000 touchpoints for 1,000 users.

### Dataset Access
- **Download**: For convenience, you can view the dataset directly from [this Google Drive link](https://drive.google.com/file/d/106-IrB8nvlNx7KynqSTrQjkNu2ldz79t/view?usp=sharing).

### Channel Definitions
In this context:
- **Display**: Refers to display advertising - visual ads shown on websites (banners, images, etc.)
- **Video**: Video advertising on platforms like YouTube or streaming services
- **Social**: Marketing efforts on social media platforms
- **Search**: Paid or organic search engine marketing
- **Email**: Email marketing campaigns
- **Affiliate**: Marketing through affiliate partners and referral networks

## Implemented Attribution Models

### 1. First-Touch Attribution
Attributes 100% of conversion credit to the first touchpoint in the customer journey.
```python
def first_touch_attribution(df):
    first_touches = df[df['is_first_touch']]['touchpoint'].value_counts()
    return calculate_attribution(first_touches.to_dict())
```

### 2. Last-Touch Attribution
Attributes 100% of conversion credit to the final touchpoint before conversion.
```python
def last_touch_attribution(df):
    last_touches = df[df['is_last_touch']]['touchpoint'].value_counts()
    return calculate_attribution(last_touches.to_dict())
```

### 3. Linear Attribution
Distributes credit evenly across all touchpoints in the customer journey.
```python
def linear_attribution(df):
    all_touches = df['touchpoint'].value_counts()
    return calculate_attribution(all_touches.to_dict())
```

### 4. U-shaped Attribution
Assigns 40% credit to both first and last touchpoints, with the remaining 20% distributed among middle touchpoints.
```python
def u_shaped_attribution(df):
    attribution = {}
    for _, user_journey in df.groupby('user_id'):
        journey_length = len(user_journey)
        if journey_length == 1:
            touchpoint = user_journey['touchpoint'].iloc[0]
            attribution[touchpoint] = attribution.get(touchpoint, 0) + 1
        else:
            first_touch = user_journey['touchpoint'].iloc[0]
            last_touch = user_journey['touchpoint'].iloc[-1]
            attribution[first_touch] = attribution.get(first_touch, 0) + 0.4
            attribution[last_touch] = attribution.get(last_touch, 0) + 0.4

            middle_weight = 0.2 / (journey_length - 2) if journey_length > 2 else 0
            for touchpoint in user_journey['touchpoint'].iloc[1:-1]:
                attribution[touchpoint] = attribution.get(touchpoint, 0) + middle_weight

    return calculate_attribution(attribution)
```

### 5. Data-Driven Attribution (Logistic Regression)
Uses logistic regression to assign credit based on statistical significance.
```python
def data_driven_attribution(X, y, touchpoints):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train logistic regression model
    model = LogisticRegression(random_state=42)
    model.fit(X_train, y_train)

    # Create attribution dictionary from model coefficients
    attribution = dict(zip(touchpoints, model.coef_[0]))
    total = sum(abs(value) for value in attribution.values())
    attribution = {channel: abs(value) / total for channel, value in attribution.items()}

    return attribution, model.score(X_test, y_test)
```

### 6. Random Forest Attribution
Leverages tree-based machine learning to identify complex patterns and interactions.
```python
def random_forest_attribution(X, y, label_encoder):
    # Split the data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Train Random Forest model
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
    rf_model.fit(X_train, y_train)
    
    # Calculate attribution from feature importances
    attribution = {}
    channel_importances = rf_model.feature_importances_[:len(label_encoder.classes_)]
    
    for channel, importance in zip(label_encoder.classes_, channel_importances):
        attribution[channel] = importance
    
    # Normalize attribution
    total = sum(attribution.values())
    attribution = {channel: value / total for channel, value in attribution.items()}
    
    return attribution, accuracy_score(y_test, rf_model.predict(X_test))
```

## Project Structure

The project follows a standard data science workflow:

1. **Data Preprocessing**: Cleaning, sorting, and structuring the data
2. **Exploratory Data Analysis (EDA)**: Understanding data distributions and patterns
3. **Feature Engineering**: Creating representations of user journeys for modeling
4. **Model Implementation**: Building and running attribution models
5. **Evaluation & Comparison**: Comparing results across different models
6. **Visualization**: Presenting results visually for interpretation
7. **Business Insights**: Deriving actionable marketing strategy recommendations

## 7. Business Insights and Applications

### First-Touch Attribution
- **Business Application**: Customer acquisition strategy optimization
- **Implementation**: Allocate awareness campaign budget to high-performing first-touch channels
- **Key Insight**: Display and video channels are most effective at introducing new customers to the brand

### Last-Touch Attribution
- **Business Application**: Conversion rate optimization
- **Implementation**: Focus conversion-oriented content on top last-touch channels
- **Key Insight**: Search and email are most effective at closing sales

### Linear Attribution
- **Business Application**: Balanced channel investment strategy
- **Implementation**: Ensure consistent messaging and quality across all touchpoints
- **Key Insight**: Video has slightly higher overall influence across the customer journey

### U-shaped Attribution
- **Business Application**: Dual-focus funnel optimization (awareness + conversion)
- **Implementation**: Invest 80% of resources in first and last touchpoints, 20% in middle interactions
- **Key Insight**: Display ads excel at both introducing customers and supporting conversions

### Data-Driven Attribution (Logistic Regression)
- **Business Application**: ROI optimization based on statistical evidence
- **Implementation**: Allocate budget proportionally to attribution weights derived from data
- **Key Insight**: Display, social, and search have the strongest statistical relationship with conversion

### Random Forest Attribution
- **Business Application**: Complex pattern discovery and interaction analysis
- **Implementation**: Identify and leverage channel synergies based on tree-based importance
- **Key Insight**: Video has dramatically higher importance when considering complex interactions

### Integrated Marketing Strategy
By combining insights from all six models, businesses can develop a comprehensive marketing strategy that:

1. **Optimizes Channel Budget Allocation**: Directing resources based on each channel's role and importance
2. **Tailors Content by Channel Purpose**: Creating awareness content for first-touch channels and conversion content for last-touch channels
3. **Improves Customer Journey Design**: Structuring the optimal sequence of touchpoints based on ML model insights
4. **Maximizes Marketing ROI**: Focusing investment on channels with highest impact at each funnel stage
5. **Enables Data-Driven Decision Making**: Replacing gut-feel marketing decisions with evidence-based strategy

## Technologies Used

- **Python**: Primary programming language
- **Pandas & NumPy**: Data manipulation and numerical operations
- **Scikit-learn**: Machine learning model implementation
- **Matplotlib & Seaborn**: Data visualization
- **Google Colab**: Development environment


## Future Improvements

- **Real Data Integration**: Extend to work with real marketing data from Google Analytics or Adobe Analytics
- **Time Decay Model**: Add time-based attribution weighting
- **Interactive Dashboard**: Create interactive visualization dashboard for exploring results
- **Segment Analysis**: Add capability to analyze attribution by customer segments

## Conclusion

MarketPath Analytics demonstrates the value of applying multiple attribution models to understand marketing effectiveness. By comparing different approaches, marketers can gain a more holistic view of how channels contribute to conversions, leading to more informed budget allocation and campaign optimization decisions.
