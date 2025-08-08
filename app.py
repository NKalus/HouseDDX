from PIL import Image
import tensorflow as tf
tf.config.set_visible_devices([], 'GPU')  # Force CPU-only mode on Render
import os
import random
import numpy as np
from flask import Flask, render_template, request
from keras.applications.mobilenet_v2 import MobileNetV2, preprocess_input, decode_predictions
from keras.preprocessing import image
from tensorflow.keras.applications import MobileNetV2
import traceback

# --- Environment variables to prevent TensorFlow GPU issues on macOS ---
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ["TF_ENABLE_ONEDNN_OPTS"] = "0"
os.environ["PROTOCOL_BUFFERS_PYTHON_IMPLEMENTATION"] = "python"

# --- Flask setup ---
app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# --- Load pre-trained model ---
model = MobileNetV2(weights='imagenet')

# --- Diagnosis bank ---
diagnosis_bank = {
    "rash": [
        "Hyperreactive dermoepidermal exuberance syndrome (a.k.a. your skin is being dramatic)",
        "Vesiculopathic keratinocyte rebellion (your skin cells unionized and went on strike)",
        "Transient necrotizing flambodermatitis (because normal rashes are too mainstream)",
        "Autoimmune blistering with photophobic angst (edgy and light-sensitive)",
        "Idiopathic itchosaurus complex (science meets imaginary dinosaur)",
        "Contact-induced stupidity pox (possibly from hugging people who say 'I'm not a doctor, but...')",
        "Subdermal drama eruption (your skin can't handle your personality)",
        "Dermal overreaction syndrome (may be contagious on social media)",
        "Multifocal epidermal tantrum disorder (every pore is throwing a fit)",
        "Heat-seeking dermatitis with zero chill",
        "Emotionally unstable skin syndrome (diagnosed by your ex)",
        "Pseudoallergic dermal flailing (an allergic reaction to your own BS)",
        "Neurotic scratch cycle disorder (itch, scratch, regret, repeat)",
        "Localized hypersensitivity to your own nonsense",
        "Frantic blister regression disorder (your skin tried to cancel itself)"
    ],
    "swelling": [
        "Inflammatory bloatosis maximus (your tissues are auditioning for a Michelin Man reboot)",
        "Interstitial idiopathic puffification disorder (cause: unknown, swelling: undeniable)",
        "Tissue rebellion syndrome (they've unionized and demand less sodium)",
        "Chronic edematosis of the hypochondrium (you're 40% fluid, 60% paranoia)",
        "Pseudolymphatic egomania (your ego is retaining water)",
        "Ballooning of the moron lobe (not a tumor, just a poor life choice made physical)",
        "Swellomatosis of the facial embarrassment zone",
        "Aquapocalyptic infiltration disorder (you're drowning from the inside out)",
        "Venolymphatic ego retention (the arrogance won’t drain either)",
        "Passive-aggressive fluid accumulation",
        "Hydrospheric entitlement edema",
        "Inflamed self-worth sacs (diagnosed after checking your Twitter)",
        "Psychosocial bloat disorder",
        "Unsolicited opinion-based tissue expansion",
        "Whiny gland hypertrophy"
    ],
    "bruise": [
        "Contusional dumbassity (obtained by walking into a 'Pull' door marked 'Push')",
        "Capillary fragility of the terminally clumsy",
        "Ecchymotic memory recall lesion (a bruise triggered by remembering your ex)",
        "Iatrogenic clumsopathy (self-inflicted, but you’ll still blame the doctor)",
        "Blunt force irony syndrome",
        "Spontaneous thrombopurpuric embarrassment",
        "Shame compression under skin (emotional bruising now visible to all)",
        "Denial-induced discolorative patterning",
        "Platelet-induced regret patch",
        "Vascular apology formation (the bruise says what words never will)",
        "Drama imprint syndrome",
        "Purpura of avoidable stupidity",
        "Trauma seepage of the brainless variety",
        "Sympathy contusion cluster",
        "Impulsive chaos bruise"
    ],
    "growth": [
        "Benign attention-seeking pseudotumor (it’s not dangerous, just annoying—like you)",
        "Narcissistic nodularity (diagnosed via selfie frequency)",
        "Mesenchymal unresolved-issue mass",
        "Superiority complex encapsulated in soft tissue",
        "Proliferation of smug cells",
        "Overcompensation fibroid",
        "Delusionoma, slow-growing but terminal at parties",
        "Hope cyst with a low probability of fulfillment",
        "Cognitive tumor of bad decisions",
        "Self-diagnosed reality lump",
        "Overconfidence blob (resistant to feedback)",
        "Unbiopsyable meat mystery (not FDA approved)",
        "Stagnant ambition carcinoma",
        "Ego-noma of the low self-esteem quadrant",
        "Nonpalpable potential node"
    ],
    "burn": [
        "First-degree idiocy scorch (because you touched it after saying 'I think it's cooled down')",
        "Consequential thermogenic karma",
        "Superficial ego-melting rash",
        "Dermal nuking via kitchen curiosity",
        "Pyrodramatitis (ignited by emotional combustion)",
        "Open-flame barbeculopathy",
        "Steam-powered blister bloom",
        "Sarcasm-induced keratinolysis",
        "Mild stupidity singe",
        "Sunburn of the chronically unprepared",
        "SPF-0 decision blister",
        "Microwave hubris syndrome",
        "Heartbreak-induced thermal inflammation",
        "Toaster duel fallout",
        "Curiosity burn unit admission pending"
    ],
    "lump": [
        "Lump of existential dread (hard to the touch, harder to explain)",
        "Guilt gland hyperplasia",
        "Bump of unresolved baggage",
        "Localized ambition avoidance cyst",
        "Tactile regret deposit",
        "Encapsulated WTFoma (we don't know either)",
        "Blob of medical confusion",
        "Spontaneous disappointment tumor",
        "Oopsie glandular prolapse",
        "Self-worth pseudonodule",
        "Embarrassoma (it's watching you back)",
        "Procrastination growth, chronic",
        "Meatball of metaphysical concern",
        "Psychosomatic flesh bundle",
        "Snark-induced granuloma"
    ],
    "idiocy": [
        "Cerebral decelerosis (your thoughts run Windows 95)",
        "Acute moronitis with chronic backpedaling",
        "Smugnosia terminalis (inability to shut up despite zero knowledge)",
        "Reality rejection flare-up (triggered by facts)",
        "Cortical slowdown with live-commentary syndrome",
        "Neurohumoral collapse caused by confidence in conspiracy theories",
        "Post-truth lobular erosion",
        "Flatline of insight with TikTok overlays",
        "Meme-induced logical infarct",
        "Memory hole echo chamber disorder",
        "Dunning-Krugerosis, advanced stage",
        "Terminal intellect bypass (brain online, user offline)",
        "Logical non-compliance syndrome",
        "Hippocampal inflammation due to Reddit overdose",
        "Recurring psychosomatic derp",
        "Cortical smug overload",
        "Hyperactive B.S. synthesis gland",
        "Moronic viral content exposure",
        "One-sided logic deflation",
        "Misinformational brain rot"
    ]
}

def classify_image(img_path):
    img = image.load_img(img_path, target_size=(224, 224))
    x = image.img_to_array(img)
    x = np.expand_dims(x, axis=0)
    x = preprocess_input(x)
    preds = model.predict(x)
    label = decode_predictions(preds, top=1)[0][0][1]
    return label.lower()

def map_to_tag(label):
    if random.random() < 0.07:
        return "idiocy"
    if any(word in label for word in ["rash", "eczema", "blister", "hives"]):
        return "rash"
    elif any(word in label for word in ["bruise", "contusion", "hematoma"]):
        return "bruise"
    elif any(word in label for word in ["swelling", "edema", "inflammation"]):
        return "swelling"
    elif any(word in label for word in ["lump", "tumor", "mass", "growth"]):
        return "growth"
    elif any(word in label for word in ["burn", "scald", "char"]):
        return "burn"
    else:
        return random.choice(list(diagnosis_bank.keys()))

def generate_diagnosis(img_path):
    label = classify_image(img_path)
    tag = map_to_tag(label)
    return random.choice(diagnosis_bank[tag])

@app.route('/', methods=['GET', 'POST'])
def index():
    diagnosis = None
    quote = None
    quotes = [
        "Everybody lies.",
        "I take risks. Sometimes patients die. But not taking risks causes more patients to die — so I guess my biggest problem is I've been cursed with the ability to do the math.",
        "It's never lupus. Except when it is.",
        "Humanity is overrated.",
        "If you talk to God, you're religious. If God talks to you, you're psychotic.",
        "I'm not always miserable. Sometimes I'm asleep."
    ]

    if request.method == 'POST' and 'image' in request.files:
        file = request.files['image']
        if file.filename:
            filepath = os.path.join(app.config['UPLOAD_FOLDER'], file.filename)
            file.save(filepath)
            diagnosis = generate_diagnosis(filepath)
            quote = random.choice(quotes)

    return render_template('index.html', diagnosis=diagnosis, random_quote=quote)

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5050))
    app.run(host="0.0.0.0", port=port, debug=True)
