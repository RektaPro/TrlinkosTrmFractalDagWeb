import requests

def download_data(url, output_file):
    """
    Télécharge un fichier depuis une URL et l'enregistre localement.

    :param url: L'URL du fichier à télécharger.
    :param output_file: Le chemin du fichier de sortie.
    """
    try:
        response = requests.get(url)
        response.raise_for_status()
        with open(output_file, "wb") as file:
            file.write(response.content)
        print(f"Données téléchargées et enregistrées sous {output_file}")
    except requests.exceptions.RequestException as e:
        print(f"Erreur lors du téléchargement : {e}")

if __name__ == "__main__":
    # Remplacez par l'URL des données et le chemin de sortie souhaité
    url = "https://google.com"
    output_file = "google_homepage.html"

    download_data(url, output_file)
