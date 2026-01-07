from app.schemas.meal import WeeklyMealResponse


def plan_weekly_meal() :
    return {
        "week":{
            "Monday":{
            "breakfast": ["Oatmeal", "Boiled Egg"],
            "lunch": ["Stir-fried Cabbage with Pork"],
            "dinner": ["Radish Soup"]
        },
        "Tuesday": {
            "breakfast": ["Soy Milk", "Bun"],
            "lunch": ["Broccoli Stir-fry"],
            "dinner": ["Tomato Egg Soup"]
        }
    }
    }
