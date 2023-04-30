import os
from datetime import datetime, timedelta
from typing import List

import motor.motor_asyncio
import pandas as pd
from bson import ObjectId
from fastapi import FastAPI, Request, Body
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# show all columns
pd.set_option('display.max_columns', None)

app = FastAPI()

print("Starting server...")

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# load environment variables
from dotenv import load_dotenv

load_dotenv()

client = motor.motor_asyncio.AsyncIOMotorClient(os.environ["MONGODB_URL"])
db = client.food_restaturent

data_length = 100000

from recommand import get_rec
from sentiment import train_sentiment, predict_sentiment


class PyObjectId(ObjectId):
    @classmethod
    def __get_validators__(cls):
        yield cls.validate

    @classmethod
    def validate(cls, v):
        if not ObjectId.is_valid(v):
            raise ValueError("Invalid objectid")
        return ObjectId(v)

    @classmethod
    def __modify_schema__(cls, field_schema):
        field_schema.update(type="string")


# const feedbackSchema = new Schema(
#     feedback: { type: String },
#     userID: { type: Schema.Types.ObjectId, ref: "User" },
#     orderId: { type: Schema.Types.ObjectId, ref: "Order" },
#     feedback_text: { type: String },
#   { collection: "feedbacks" }

class FeedbackModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    userID: PyObjectId = Field(default_factory=PyObjectId)
    orderID: PyObjectId = Field(default_factory=PyObjectId)
    feedbackdetils: str
    rateValue: str
    sentiment: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "userID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "orderID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "feedbackdetils": "This is a test feedback",
                "rateValue": 4,
                "sentiment": "neutral",
            }
        }


@app.get("/feedbacks", response_description="List all feedbacks", response_model=List[FeedbackModel])
async def list_feedbacks():
    feedbacks = await db["feedbacks"].find().to_list(data_length)
    return feedbacks


# find by user id
@app.get("/feedbacks/{id}", response_description="Get a single feedback", response_model=FeedbackModel)
async def show_feedback(id: str):
    # get feedbackdetils by id
    feedback = await db["feedbacks"].find_one({"_id": ObjectId(id)})
    if feedback:
        return feedback

    return {"error": "feedback not found"}


# feedback update model, only update sentiment
class FeedbackUpdateModel(BaseModel):
    sentiment: str = Field(...)

    class Config:
        schema_extra = {
            "example": {
                "sentiment": "positive",
            }
        }

# update feedback sentiment
@app.put("/feedbacks/{id}", response_description="Update a feedback", response_model=FeedbackModel)
async def update_feedback(id: str, feedback: FeedbackUpdateModel = Body(...)):
    # update sentiment by id
    feedback = await db["feedbacks"].update_one({"_id": id}, {"$set": {"sentiment": feedback.sentiment}})
    if feedback:
        return feedback

    return {"error": "feedback not found"}


# const foodCategoriesSchema = new Schema(
#     _id: { type: Schema.Types.ObjectId},
#     description: { type: String },
#     image: { type: String },
#     name: { type: String },
#   { collection: "foodCategories" }

class foodCategoriesModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    description: str = Field(...)
    image: str = Field(...)
    name: str = Field(...)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c",
                "description": "Burger with cheese",
                "image": "https://www.food.com",
                "name": "Burger"
            }
        }


# List all food categories
@app.get("/foodCategories", response_description="List all food categories", response_model=List[foodCategoriesModel])
async def list_food_categories():
    foodCategories = await db["foodCatergories"].find().to_list(data_length)
    return foodCategories


# const foodsSchema = new Schema(
#     _id: { type: Schema.Types.ObjectId, ref: "User" },
#     name: { type: String },
#     price: { type: Number },
#     description: { type: String },
#     image: { type: String },
#     category: { type: Schema.Types.ObjectId, ref: "FoodCategory" },},
#   { collection: "foods" }

class foodsModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    name: str = Field(...)
    price: int = Field(...)
    description: str = Field(...)
    image: str = Field(...)
    category: PyObjectId = Field(default_factory=PyObjectId)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c",
                "name": "Burger",
                "price": 100,
                "description": "Burger with cheese",
                "image": "https://www.food.com",
                "category": "60f4d5c5b5f0f0e5e8b2b5c"
            }
        }


# List all foods
@app.get("/foods", response_description="List all foods", response_model=List[foodsModel])
async def list_foods():
    foods = await db["foods"].find().to_list(data_length)
    return foods


# const orderItemsSchema = new Schema(
#     orderID: { type: Schema.Types.ObjectId, ref: "Order" },
#     food: { type: Schema.Types.ObjectId, ref: "Food" },
#     quantity: { type: Number },
#     price: { type: Number },
#   { collection: "orderItemWithQuantities" }


class OrderItemWithQuantityModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    orderID: PyObjectId = Field(default_factory=PyObjectId)
    food: PyObjectId = Field(default_factory=PyObjectId)
    price: int
    quantity: int

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "_id": "60f4d5c5b5f0f0e5e8b2b5c9",
                "orderID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "food": "60f4d5c5b5f0f0e5e8b2b5c9",
                "price": 20,
                "quanitity": 2
            }
        }


@app.get(
    "/orderItemWithQuantities", response_description="List all order items with quantities",
    response_model=List[OrderItemWithQuantityModel]
)
async def list_order_items_with_quantities():
    order_items_with_quantities = await db["orderItemWithQuantities"].find().to_list(data_length)
    return order_items_with_quantities


# const orderSchema = new Schema(
#     createDate: { type: String },
#     createTime: { type: String },
#     status: { type: String },
#     orderedBy: { type: Schema.Types.ObjectId, ref: "User" },
#     billValue: { type: String },
#     discount: { type: String },
#     orderType: { type: Schema.Types.ObjectId, ref: "OrderType" },
#     table: { type: Schema.Types.ObjectId, ref: "Table" },
#     handleBy: { type: Schema.Types.ObjectId, ref: "Employee" },
#   { collection: "orders" }


class OrderModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    createDate: str = Field(...)
    createTime: str = Field(...)
    status: str = Field(...)
    orderedBy: PyObjectId = Field(default_factory=PyObjectId)
    billValue: str = Field(...)
    discount: str = Field(...)
    orderType: str = Field(...)
    table: str = Field(...)
    handleBy: str = Field(...)

    # orderType: PyObjectId = Field(default_factory=PyObjectId)
    # table: PyObjectId = Field(default_factory=PyObjectId)
    # handleBy: PyObjectId = Field(default_factory=PyObjectId)

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "createDate": "2021-07-21",
                "createTime": "18:11:10",
                "status": "pending",
                "orderedBy": "60f4d5c5b5f0f0e5e8b2b5c9",
                "billValue": "100",
                "discount": "10",
                "orderType": "60f4d5c5b5f0f0e5e8b2b5c9",
                "table": "60f4d5c5b5f0f0e5e8b2b5c9",
                "handleBy": "60f4d5c5b5f0f0e5e8b2b5c9"
            }
        }


@app.get("/orders", response_description="List all orders", response_model=List[OrderModel])
async def list_orders():
    orders = await db["orders"].find().to_list(data_length)
    return orders


# const userSchema = new Schema(
#     userID: { type: Schema.Types.ObjectId, ref: "User" },
#     firstName: { type: String },
#     lastName: { type: String },
#     userName: { type: String },
#     email: { type: String },
#     dateOfBirth: { type: String },
#     mobileNumber: { type: String },
#     password: { type: String },
#   { collection: "users" }

class UserModel(BaseModel):
    id: PyObjectId = Field(default_factory=PyObjectId, alias="_id")
    firstName: str
    lastName: str
    userName: str
    email: str
    dateOfBirth: str
    mobileNumber: str
    password: str

    class Config:
        allow_population_by_field_name = True
        arbitrary_types_allowed = True
        json_encoders = {ObjectId: str}
        schema_extra = {
            "example": {
                "userID": "60f4d5c5b5f0f0e5e8b2b5c9",
                "firstName": "John",
                "lastName": "Doe",
                "userName": "JohnDoe",
                "email": "johndoe@example.com",
                "dateOfBirth": "1990-01-01",
                "mobileNumber": "0712345678",
                "password": "password"
            }
        }


@app.get("/users", response_description="List all users", response_model=List[UserModel])
async def list_users():
    users = await db["users"].find().to_list(data_length)
    return users


# 1. Aggregate orders with orderItemWithQuantities
# 2. for each orderItem get orderedBy and for that get dateOfBirth from users
# 3. for each orderItem get food_id, food_type, Ingredients from foods by food_name
#
# user_id        :  orders.orderedBy
# age            :  users.dateOfBirth
# food_id        :  orderItemWithQuantities._id
# food_name      :  foods.food
# food_type      :  foods.food_type
# Ingredients    :  foods.Ingredients
# food_rating    :  feedbacks.feedback

# foodsModel
# OrderItemWithQuantityModel
# OrderModel
# UserModel
# FeedbackModel

# take data from orders, orderItemWithQuantities, foods, users, feedbacks and aggregate them
# 1. Aggregate orders with orderItemWithQuantities
# 2. for each orderItem get orderedBy and for that get dateOfBirth from users
# 3. for each orderItem get food_id, food_type, Ingredients from foods by food_name


#   {
#     "_id": "64315e59362c27c707fe156b",
#     "food_name": "Pol Roti with chili salad ",
#     "food_type": "Vegetarian",
#     "food_cuisine": "Sri Lankan",
#     "ingredients": "Flour\nCoconut\nOnion\nGreen chili\nSri Lankan spices"
#   },


#   {
#     "_id": "64316a3f362c27c707fe18a9",
#     "orderID": "64316105362c27c707fe15ec",
#     "food": "64315e59362c27c707fe1586",
#     "quantity": 2,
#     "price": 885
#   },


# {
#     "_id": "64316ada362c27c707fe20c6",
#     "createDate": "1970-10-20",
#     "createTime": "18:11:10",
#     "status": "pending",
#     "orderedBy": "64315d86362c27c707fe155c",
#     "billValue": "774",
#     "discount": "94",
#     "orderType": "1",
#     "table": "2",
#     "handleBy": "8"
#   },


# {
#     "_id": "64315d86362c27c707fe1529",
#     "firstName": "Justin",
#     "lastName": "Diaz",
#     "userName": "JustinDiaz",
#     "email": "Justin.Diaz@gmail.com",
#     "dateOfBirth": "2005-06-29",
#     "mobileNumber": "(326)149-8817",
#     "password": "v^5BXk2C"
#   },


#   {
#     "_id": "64316b59362c27c707fe2386",
#     "feedback": "2",
#     "orderID": null
#   },


base_url = "http://localhost:8000/"
save_path = "data/"


def aggregate_data():
    # read csv
    orderItemWithQuantities_df = pd.read_csv(save_path + "orderItemWithQuantities.csv")
    orders_df = pd.read_csv(save_path + "orders.csv")
    foods_df = pd.read_csv(save_path + "foods.csv")
    users_df = pd.read_csv(save_path + "users.csv")
    feedbacks_df = pd.read_csv(save_path + "feedbacks.csv")
    food_category_df = pd.read_csv(save_path + "food_categories.csv")

    # rename _id to id in all dataframes
    # orderItemWithQuantities_df.rename(columns={'_id': 'id'}, inplace=True)
    # orders_df.rename(columns={'_id': 'id'}, inplace=True)
    # foods_df.rename(columns={'_id': 'id'}, inplace=True)
    # users_df.rename(columns={'_id': 'id'}, inplace=True)
    # feedbacks_df.rename(columns={'_id': 'id'}, inplace=True)

    print(orderItemWithQuantities_df.head())
    print(orders_df.head())
    print(foods_df.head())
    print(users_df.head())
    print(feedbacks_df.head())
    print(food_category_df.head())

    # keep only required columns
    # ['_id', 'orderID', 'food']
    orderItemWithQuantities_df = orderItemWithQuantities_df[['_id', 'orderID', 'food']]
    # ['_id', 'orderedBy', 'orderType']
    orders_df = orders_df[['_id', 'orderedBy', 'orderType']]
    # ['_id', 'food_name', 'food_type', 'food_cuisine', 'ingredients']
    foods_df = foods_df[['_id', 'name', 'price', 'description', 'category']]
    # ['_id', 'dateOfBirth']
    users_df = users_df[['_id', 'dateOfBirth']]
    # ['_id', 'orderID', 'feedback']
    feedbacks_df = feedbacks_df[['_id', 'orderId', 'rateValue']]
    # rename orderId to orderID and feedbackdetils to feedback
    feedbacks_df.rename(columns={'orderId': 'orderID', 'rateValue': 'feedback'}, inplace=True)
    # ['_id', 'name']
    food_category_df = food_category_df[['_id', 'name']]

    # rename orderWithQuantities_df _id to orderItemID
    orderItemWithQuantities_df.rename(columns={'_id': 'orderItemID'}, inplace=True)

    # print list of columns in each dataframe
    print("orderItemWithQuantities_df: ", orderItemWithQuantities_df.columns)
    print("orders_df: ", orders_df.columns)
    print("foods_df: ", foods_df.columns)
    print("users_df: ", users_df.columns)
    print("feedbacks_df: ", feedbacks_df.columns)

    # add orders_df to orderWithQuantities_df by id to orderID
    df = pd.merge(orderItemWithQuantities_df, orders_df, left_on='orderID', right_on='_id')

    if '_id' in df.columns:
        print(df[df['orderID'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # add foods_df to df by id to food
    df = pd.merge(df, foods_df, left_on='food', right_on='_id')
    if '_id' in df.columns:
        print(df[df['food'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # add users_df to df by id to orderedBy
    df = pd.merge(df, users_df, left_on='orderedBy', right_on='_id')
    if '_id' in df.columns:
        print(df[df['orderedBy'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # add feedbacks_df to df by id to orderID
    df = pd.merge(df, feedbacks_df, left_on='orderID', right_on='orderID')
    if '_id' in df.columns:
        print(df[df['orderID'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    print("\n\nBefore Merge")
    # add food_category_df to df by id to category
    df = pd.merge(df, food_category_df, left_on='category', right_on='_id')
    if '_id' in df.columns:
        print(df[df['category'] != df['_id']])
        df.drop(columns=['_id'], inplace=True)

    # save to csv
    df.to_csv(save_path + "aggregate.csv", index=False)

    print("\n\nName of the file: aggregate")
    print(df.head())


def process_data():
    # load csv
    df = pd.read_csv(save_path + "aggregate.csv")

    # convert dateOfBirth to age
    df['dateOfBirth'] = pd.to_datetime(df['dateOfBirth'])
    print(df['dateOfBirth'].head())
    # now date in 2002-08-30 format
    from datetime import datetime

    now = datetime.now().strftime("%Y-%m-%d")

    df['age'] = pd.to_datetime(now) - df['dateOfBirth']
    df.drop(columns=['dateOfBirth'], inplace=True)

    # convert age to years (int)
    df['age'] = df['age'].dt.days / 365
    df['age'] = df['age'].astype(int)

    # drop orderType, price, food, category
    df.drop(columns=['orderType', 'price', 'category', 'orderID'], inplace=True)

    # rename columns name_x -> food, name_y -> cuisine
    df.rename(columns={'name_x': 'food_name', 'name_y': 'cuisine', 'food': 'food_id'}, inplace=True)

    # save to csv
    df.to_csv(save_path + "processed.csv", index=False)


def analyze_data():
    # load csv
    df = pd.read_csv(save_path + "processed.csv")

    # data types
    print(df.dtypes)

    # null values
    print(df.isnull().sum())

    # duplicate values
    print(df.duplicated().sum())

    # unique values
    print(df.nunique())

    # describe
    print(df.describe())


def pre_process():
    # load csv
    df = pd.read_csv(save_path + "processed.csv")

    # remove column if 75% of the values are null
    df.dropna(thresh=len(df) * 0.25, axis=1, inplace=True)

    # remove null values
    df.dropna(inplace=True)

    # remove null values
    df.dropna(inplace=True)

    # remove duplicate values
    df.drop_duplicates(inplace=True)

    # save to csv
    df.to_csv(save_path + "pre_processed.csv", index=False)


def update_log():
    # create txt if not exists and write to it f.write("Last updated: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    if not os.path.exists(save_path + "log.txt"):
        f = open(save_path + "log.txt", "w")
        f.close()

    # open txt and clean write to it
    f = open(save_path + "log.txt", "w")
    f.write("Last updated: " + datetime.now().strftime("%d/%m/%Y %H:%M:%S"))
    f.close()


def load_log():
    # load txt
    f = open(save_path + "log.txt", "r")
    read_data = f.read()
    print(read_data)
    f.close()

    # extract datetime
    new_datetime = read_data.split("Last updated: ")[1]

    # convert to datetime
    new_datetime = datetime.strptime(new_datetime, "%d/%m/%Y %H:%M:%S")

    # if last update was more than 1 hour ago
    if datetime.now() - new_datetime > timedelta(hours=1):
        return True
    else:
        return False


# Recommendation
@app.get("/recommendation/{user_id}")
async def get_recommendation(user_id: str, num_of_rec: int = 5):
    if num_of_rec:
        recommendation, user_stat = get_rec(user_id, num_of_rec=num_of_rec)
    else:
        recommendation, user_stat = get_rec(user_id, num_of_rec=5)

    update_log()
    return {"recommendations": recommendation}


# Load data and Recommendation
@app.get("/recommendation-load/{user_id}")
async def get_recommendation_load(user_id: str, num_of_rec: int = 5):
    orderItemWithQuantities = await list_order_items_with_quantities()
    orders = await list_orders()
    foods = await list_foods()
    users = await list_users()
    feedbacks = await list_feedbacks()
    food_categories = await list_food_categories()

    orderItemWithQuantities = pd.DataFrame(orderItemWithQuantities)
    orders = pd.DataFrame(orders)
    foods = pd.DataFrame(foods)
    users = pd.DataFrame(users)
    feedbacks = pd.DataFrame(feedbacks)
    food_categories = pd.DataFrame(food_categories)

    orderItemWithQuantities.to_csv(save_path + "orderItemWithQuantities.csv", index=False)
    orders.to_csv(save_path + "orders.csv", index=False)
    foods.to_csv(save_path + "foods.csv", index=False)
    users.to_csv(save_path + "users.csv", index=False)
    feedbacks.to_csv(save_path + "feedbacks.csv", index=False)
    food_categories.to_csv(save_path + "food_categories.csv", index=False)

    aggregate_data()
    process_data()
    pre_process()
    update_log()

    if num_of_rec:
        recommendation, user_stat = get_rec(user_id, num_of_rec=num_of_rec)
    else:
        recommendation, user_stat = get_rec(user_id, num_of_rec=5)
    return {"recommendations": recommendation}


# Load data
@app.get("/load-data")
async def load_data(request: Request):
    orderItemWithQuantities = await list_order_items_with_quantities()
    orders = await list_orders()
    foods = await list_foods()
    users = await list_users()
    feedbacks = await list_feedbacks()
    food_categories = await list_food_categories()

    orderItemWithQuantities = pd.DataFrame(orderItemWithQuantities)
    orders = pd.DataFrame(orders)
    foods = pd.DataFrame(foods)
    users = pd.DataFrame(users)
    feedbacks = pd.DataFrame(feedbacks)
    food_categories = pd.DataFrame(food_categories)

    orderItemWithQuantities.to_csv(save_path + "orderItemWithQuantities.csv", index=False)
    orders.to_csv(save_path + "orders.csv", index=False)
    foods.to_csv(save_path + "foods.csv", index=False)
    users.to_csv(save_path + "users.csv", index=False)
    feedbacks.to_csv(save_path + "feedbacks.csv", index=False)
    food_categories.to_csv(save_path + "food_categories.csv", index=False)

    aggregate_data()
    process_data()
    pre_process()
    update_log()

    return {"status": "success"}


# Load data if last update is more than 1 hour and Recommendation
@app.get("/recommendation-load-update/{user_id}")
async def get_recommendation_load_update(user_id: str, num_of_rec: int = 5):
    if load_log():
        orderItemWithQuantities = await list_order_items_with_quantities()
        orders = await list_orders()
        foods = await list_foods()
        users = await list_users()
        feedbacks = await list_feedbacks()
        food_categories = await list_food_categories()

        orderItemWithQuantities = pd.DataFrame(orderItemWithQuantities)
        orders = pd.DataFrame(orders)
        foods = pd.DataFrame(foods)
        users = pd.DataFrame(users)
        feedbacks = pd.DataFrame(feedbacks)
        food_categories = pd.DataFrame(food_categories)

        orderItemWithQuantities.to_csv(save_path + "orderItemWithQuantities.csv", index=False)
        orders.to_csv(save_path + "orders.csv", index=False)
        foods.to_csv(save_path + "foods.csv", index=False)
        users.to_csv(save_path + "users.csv", index=False)
        feedbacks.to_csv(save_path + "feedbacks.csv", index=False)
        food_categories.to_csv(save_path + "food_categories.csv", index=False)

        aggregate_data()
        process_data()
        pre_process()
        update_log()

    if num_of_rec:
        recommendation, user_stat = get_rec(user_id, num_of_rec=num_of_rec)
    else:
        recommendation, user_stat = get_rec(user_id, num_of_rec=5)
    return {"recommendations": recommendation}


# sentiment analysis
@app.get("/sentiment-analysis/{text}")
async def get_sentiment_analysis(text: str):
    sentiment = predict_sentiment(text)
    return {"sentiment": sentiment}


# train sentiment model
@app.get("/train-sentiment-model")
async def train_sentiment_model():
    train_sentiment()
    return {"status": "success"}


# update user feedback
@app.post("/update-feedback/{order_id}")
async def update_feedback(feedback_id: str):
    try:
        feedback = await show_feedback(feedback_id)
        sentiment = predict_sentiment(feedback["feedbackdetils"])
        feedback = await db["feedbacks"].update_one({"_id": ObjectId(feedback["_id"])}, {"$set": {"sentiment": sentiment}})

        return {"sentiment": sentiment}

    except Exception as e:
        print(e)
        return {"status": "failed"}
