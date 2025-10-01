from flask import Flask, render_template, request
from sentence_transformers import SentenceTransformer, util

app = Flask(__name__)
model = SentenceTransformer('all-mpnet-base-v2')  # Modelo mais preciso
LIMIAR_MINIMO = 0.7  # Limiar mínimo mais rigoroso para considerar palavras relacionadas

@app.route("/", methods=["GET", "POST"])
def index():
    similaridade = None
    interpretacao = ""
    vetor1 = []
    vetor2 = []
    palavra1 = ""
    palavra2 = ""

    if request.method == "POST":
        palavra1 = request.form.get("palavra1").strip()
        palavra2 = request.form.get("palavra2").strip()

        # Validação básica
        if not palavra1 or not palavra2:
            interpretacao = "Por favor, insira ambas as palavras."
        else:
            # Criar frases com contexto mais específico para melhor precisão
            frase1 = f"O termo '{palavra1}' se refere a algo específico."
            frase2 = f"O termo '{palavra2}' se refere a algo específico."

            # Gerar vetores
            vetor1 = model.encode(frase1).tolist()
            vetor2 = model.encode(frase2).tolist()

            # Calcular similaridade
            similaridade = util.cos_sim(vetor1, vetor2).item()

            # Interpretação mais rigorosa e granular
            if similaridade >= 0.85:
                interpretacao = "Palavras extremamente similares ou sinônimas"
            elif similaridade >= 0.75:
                interpretacao = "Palavras muito próximas no significado"
            elif similaridade >= LIMIAR_MINIMO:
                interpretacao = "Palavras moderadamente relacionadas"
            elif similaridade >= 0.5:
                interpretacao = "Palavras com pouca relação"
            elif similaridade >= 0.3:
                interpretacao = "Palavras quase sem relação"
            else:
                interpretacao = "Palavras completamente diferentes e sem relação"

    return render_template("index.html",
                           similaridade=similaridade,
                           interpretacao=interpretacao,
                           vetor1=vetor1,
                           vetor2=vetor2,
                           palavra1=palavra1,
                           palavra2=palavra2)

if __name__ == "__main__":
    app.run(debug=True)
