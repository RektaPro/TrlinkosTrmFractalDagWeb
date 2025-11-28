import requests
from bs4 import BeautifulSoup
import time
import json
import argparse

def google_scrape(query, num_results=10):
    """
    Effectue une recherche Google et retourne les résultats.

    :param query: La requête de recherche.
    :param num_results: Nombre de résultats à retourner.
    :return: Liste des résultats de recherche.
    """
    headers = {
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36"
    }
    search_url = f"https://www.google.com/search?q={query.replace(' ', '+')}&num={num_results}"
    response = requests.get(search_url, headers=headers)

    if response.status_code == 200:
        soup = BeautifulSoup(response.text, "html.parser")
        results = []
        for g in soup.find_all('div', class_='tF2Cxc'):
            title = g.find('h3').text if g.find('h3') else "Titre non disponible"
            link = g.find('a')['href'] if g.find('a') else "Lien non disponible"
            snippet = g.find('span', class_='aCOpRe').text if g.find('span', class_='aCOpRe') else "Description non disponible"
            results.append({"title": title, "link": link, "snippet": snippet})
        return results
    else:
        print(f"Erreur : Impossible d'accéder à Google (Code {response.status_code})")
        return []

def save_results_to_file(results, filename="results.json"):
    """
    Sauvegarde les résultats dans un fichier JSON.

    :param results: Liste des résultats de recherche.
    :param filename: Nom du fichier de sortie.
    """
    with open(filename, "w", encoding="utf-8") as file:
        json.dump(results, file, ensure_ascii=False, indent=4)
    print(f"Résultats sauvegardés dans {filename}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Scraper Google en temps réel.")
    parser.add_argument("query", type=str, help="Requête de recherche Google.")
    parser.add_argument("--num_results", type=int, default=10, help="Nombre de résultats à retourner.")
    parser.add_argument("--output", type=str, default="results.json", help="Fichier de sortie pour les résultats.")
    args = parser.parse_args()

    # Effectuer la recherche
    search_results = google_scrape(args.query, num_results=args.num_results)

    # Afficher les résultats
    for result in search_results:
        print(f"Titre: {result['title']}")
        print(f"Lien: {result['link']}")
        print(f"Description: {result['snippet']}")
        print("-" * 80)

    # Sauvegarder les résultats
    save_results_to_file(search_results, filename=args.output)

    # Ajouter un délai pour éviter d'être bloqué
    time.sleep(2)
