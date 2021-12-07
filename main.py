#imports
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

#functional imports
from movie2vec import get_movies

app = FastAPI()

origins = ["*"]

app.add_middleware(
    CORSMiddleware,
    allow_origins=origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

class UserIn(BaseModel):
    movie_desc: str = ""

@app.get("/")
async def debug():
    return "Hello World"

@app.post("/get_movies")
async def movies_list(movies: UserIn):
    return get_movies(movies.movie_desc)