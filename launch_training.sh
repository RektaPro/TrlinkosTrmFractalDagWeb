#!/bin/bash
# Script shell pour lancer l'entraînement du système T-RLINKOS TRM++
# Usage: ./launch_training.sh [options passées à launch_training.py]

set -e

# Couleurs pour le terminal
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}T-RLINKOS TRM++ - Launch Training${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""

# Vérifier que Python est installé
if ! command -v python3 &> /dev/null && ! command -v python &> /dev/null; then
    echo -e "${RED}✗ Erreur: Python n'est pas installé${NC}"
    exit 1
fi

# Utiliser python3 si disponible, sinon python
PYTHON_CMD="python3"
if ! command -v python3 &> /dev/null; then
    PYTHON_CMD="python"
fi

echo -e "${GREEN}✓ Python trouvé: $($PYTHON_CMD --version)${NC}"

# Vérifier que les dépendances sont installées
echo -e "${BLUE}Vérification des dépendances...${NC}"
if ! $PYTHON_CMD -c "import torch" 2>/dev/null; then
    echo -e "${YELLOW}⚠ PyTorch n'est pas installé${NC}"
    echo -e "${YELLOW}  Installation recommandée: pip install torch${NC}"
    echo ""
    read -p "Continuer quand même? (y/N) " -n 1 -r
    echo
    if [[ ! $REPLY =~ ^[Yy]$ ]]; then
        exit 1
    fi
else
    echo -e "${GREEN}✓ PyTorch installé${NC}"
fi

echo ""

# Lancer le script Python avec tous les arguments passés
echo -e "${BLUE}Lancement de l'entraînement...${NC}"
echo ""

$PYTHON_CMD launch_training.py "$@"

exit_code=$?

echo ""
if [ $exit_code -eq 0 ]; then
    echo -e "${GREEN}✓ Entraînement terminé avec succès!${NC}"
else
    echo -e "${RED}✗ L'entraînement a échoué (code: $exit_code)${NC}"
fi

exit $exit_code
