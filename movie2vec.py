def get_movies(movie_desc):
    if(len(movie_desc)!=0):
        return {
            "name": "Best Movie",
            "genre": "action",
            "other": "Other details"
        }
    else:
        return "Error! No Description Given!"
    