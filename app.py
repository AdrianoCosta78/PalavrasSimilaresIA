from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-mpnet-base-v2')  # Modelo mais preciso

@app.route("/", methods=["GET", "POST"])
def index():
    similaridade = None
    interpretacao = ""
    vetor1 = []
    vetor2 = []
    palavra1 = ""
    palavra2 = ""

    if request.method == "POST":
        palavra1 = request.form.get("palavra1")
        palavra2 = request.form.get("palavra2")

        # Criar frases simples para contexto
        frase1 = f"Isso é sobre {palavra1}"
        frase2 = f"Isso é sobre {palavra2}"

        # Gerar vetores
        vetor1 = model.encode(frase1).tolist()
        vetor2 = model.encode(frase2).tolist()

        # Calcular similaridade
        similaridade = util.cos_sim(vetor1, vetor2).item()

        # Interpretar
        if similaridade >= 0.7:
            interpretacao = "Palavras muito próximas"
        elif similaridade >= 0.4:
            interpretacao = "Palavras moderadamente próximas"
        else:
            interpretacao = "Palavras pouco ou nada próximas"

    return render_template("index.html",
                           similaridade=similaridade,
                           interpretacao=interpretacao,
                           vetor1=vetor1,
                           vetor2=vetor2,
                           palavra1=palavra1,
                           palavra2=palavra2)

if __name__ == "__main__":
    app.run(debug=True)
