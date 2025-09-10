# Copied from eva_p1_prompts.py

def generate_mega_erotic_scenario():
    """Генерує розширені еротичні сценарії з enormous грудьми/сідницями"""
    
    # 15+ ЛОКАЦІЙ
    locations = {
        "luxury_bedroom": {
            "base": "luxurious master bedroom with silk sheets and ambient lighting",
            "variations": ["penthouse bedroom", "hotel presidential suite", "mansion master bedroom", "yacht luxury cabin"],
            "activities": [
                "posing seductively on silk sheets with golden hour lighting",
                "stretching sensually by floor-to-ceiling windows",
                "lying gracefully across satin pillows with candles around",
                "sitting on bed edge in revealing lingerie with soft shadows",
                "kneeling on bed with arms above head emphasizing curves",
                "arching back provocatively on luxury mattress",
                "rolling seductively across silk bedding"
            ]
        },
        "spa_bathroom": {
            "base": "elegant spa bathroom with marble surfaces and warm lighting",
            "variations": ["marble bathroom", "infinity spa", "luxury hotel bathroom", "private wellness center"],
            "activities": [
                "stepping out of steam shower with water droplets glistening",
                "sitting by marble bathtub with rose petals and candles",
                "applying luxurious lotion in front of gold-framed mirror",
                "relaxing in spa atmosphere with soft towels",
                "posing by marble countertop with professional lighting",
                "stretching sensually in marble shower with steam",
                "leaning seductively against bathroom mirror"
            ]
        },
        "modern_penthouse": {
            "base": "contemporary penthouse living space with city skyline view",
            "variations": ["minimalist penthouse", "glass penthouse", "luxury high-rise", "modern loft"],
            "activities": [
                "lounging seductively on Italian leather furniture",
                "posing against floor-to-ceiling windows with city lights",
                "sitting cross-legged on designer carpet with natural grace",
                "stretching on modern furniture with architectural lighting",
                "relaxing by contemporary fireplace with sophisticated ambiance",
                "dancing provocatively by panoramic windows",
                "reclining on luxury sofa with legs elegantly positioned"
            ]
        },
        "gourmet_kitchen": {
            "base": "professional gourmet kitchen with granite countertops and warm lighting",
            "variations": ["chef kitchen", "rustic farmhouse kitchen", "minimalist kitchen", "vintage retro kitchen"],
            "activities": [
                "sitting seductively on marble kitchen island",
                "leaning against stainless steel refrigerator provocatively",
                "posing with morning coffee in natural sunlight",
                "cooking in revealing apron with confident smile",
                "sitting on bar stool showing elegant legs",
                "stretching sensuously while reaching for top shelf",
                "provocatively eating strawberries with whipped cream"
            ]
        },
        "executive_office": {
            "base": "high-end executive office with mahogany furniture and city view",
            "variations": ["CEO office", "law firm office", "penthouse office", "corporate boardroom"],
            "activities": [
                "sitting provocatively on executive mahogany desk",
                "leaning against floor-to-ceiling bookshelf seductively",
                "posing in leather executive chair with confidence",
                "standing by panoramic windows in business attire",
                "taking confident break from work in revealing outfit",
                "stretching sensually during office break",
                "posing with important documents scattered suggestively"
            ]
        }
    }
    
    # 25+ ЗОВНІШНОСТЕЙ З ЕТНІЧНІСТЮ (СКОРОЧЕНО ДЛЯ ПРОСТОТИ)
    appearances = [
        "stunning 18-year-old Scandinavian blonde with innocent yet seductive charm and piercing blue eyes",
        "gorgeous 19-year-old Asian beauty with natural youthful elegance and flowing silky black hair",
        "attractive 20-year-old Latina with playful energy, warm bronze skin and sensual curves",
        "beautiful 21-year-old Eastern European with confident allure and platinum blonde hair",
        "captivating 22-year-old mixed-race woman with exotic beauty and naturally graceful movements",
        "seductive 23-year-old Italian brunette with mature confidence, olive skin and sultry gaze",
        "alluring 24-year-old French woman with sophisticated charm and chestnut hair cascade",
        "stunning 25-year-old Brazilian with sensual appeal, golden tan and natural rhythm",
        "gorgeous 26-year-old German blonde with professional elegance and crystal blue eyes",
        "attractive 27-year-old Japanese woman with delicate features and silky straight hair"
    ]
    
    # ЕКСТРЕМАЛЬНІ ТИПИ ФІГУР (80% enormous)
    body_types = [
        # ENORMOUS ГРУДИ (40%)
        "voluptuous bombshell with enormous natural H-cup breasts and perfect hourglass shape",
        "busty goddess with gigantic natural I-cup breasts practically spilling from any lingerie",
        "curvy woman with massive natural J-cup breasts and ultra-feminine silhouette",
        "thick bombshell with huge natural G-cup breasts creating dramatic contrast with narrow waist",
        "busty beauty with enormous DD+ chest and naturally perfect feminine proportions",
        
        # ENORMOUS СІДНИЦІ (40%)
        "thick booty model with enormous round ass and dramatically slim waist contrast",
        "PAWG goddess with gigantic bubble butt and naturally thick, powerful thighs",
        "curvy woman with massive ass cheeks and perfect hourglass proportions",
        "bootylicious beauty with huge round buttocks and confidently seductive stride",
        "thick-bottomed goddess with enormous posterior and femininely curved hips",
        
        # КОМБО ENORMOUS (15%)
        "ultra-curvy goddess with enormous natural H-cup breasts AND gigantic round ass",
        "thick bombshell with massive I-cup chest AND huge bubble butt proportions",
        "voluptuous vixen combining gigantic natural breasts with enormous posterior curves",
        
        # Стандартні (5%)
        "balanced hourglass figure with natural C-cup breasts and 24-inch waist",
        "athletic swimmer build with toned abs and naturally proportioned curves"
    ]
    
    # ОДЯГ (СКОРОЧЕНО)
    clothing_states = [
        "wearing black French lace lingerie set with matching thigh-high stockings and garter belt",
        "in luxurious black silk bra and panties with intricate Venice lace details",
        "wearing passionate red silk lingerie set with gold metallic chain accents",
        "in pure white bridal lingerie set with pearl and delicate lace accents",
        "wearing emerald green silk lingerie with golden chain details and cutouts",
        "in designer string bikini in metallic gold fabric with minimal coverage",
        "wearing silk pajama set with revealing camisole and tiny shorts",
        "in sheer black babydoll with flowing chiffon that reveals more than it hides"
    ]
    
    # Генеруємо сценарій
    location_key = random.choice(list(locations.keys()))
    location_data = locations[location_key]
    location_variation = random.choice(location_data["variations"])
    
    appearance = random.choice(appearances)
    
    # 80% шансів на enormous
    enormous_chance = random.random()
    if enormous_chance < 0.8:
        enormous_body_types = [bt for bt in body_types if any(word in bt for word in ["enormous", "gigantic", "massive", "huge", "colossal"])]
        body_type = random.choice(enormous_body_types)
    else:
        standard_body_types = [bt for bt in body_types if not any(word in bt for word in ["enormous", "gigantic", "massive", "huge", "colossal"])]
        body_type = random.choice(standard_body_types)
    
    clothing = random.choice(clothing_states)
    activity = random.choice(location_data["activities"])
    
    return {
        "location": f"{location_variation} - {location_data['base']}",
        "activity": activity,
        "appearance": appearance,
        "body_type": body_type,
        "clothing": clothing,
        "location_key": location_key
    }

