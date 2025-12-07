#!/usr/bin/env python3
"""
Lance l'entraînement du système T-RLINKOS TRM++

Script unifié pour lancer facilement l'entraînement du système avec
différentes configurations et modes.

Usage:
    python launch_training.py                    # Mode XOR par défaut
    python launch_training.py --mode xor         # Entraînement XOR explicite
    python launch_training.py --epochs 100       # 100 époques
    python launch_training.py --device cuda      # Utiliser GPU
    python launch_training.py --help             # Voir toutes les options
"""

import argparse
import sys


def parse_args():
    """Parse les arguments de ligne de commande."""
    parser = argparse.ArgumentParser(
        description="Lance l'entraînement du système T-RLINKOS TRM++",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Exemples d'utilisation:
  %(prog)s                               # Entraînement XOR par défaut (50 époques)
  %(prog)s --epochs 100 --lr 0.001       # Entraînement avec paramètres personnalisés
  %(prog)s --device cuda --batch-size 128  # Utiliser GPU avec batch size 128
  %(prog)s --silent                      # Mode silencieux (pas de logs détaillés)
        """
    )

    # Mode d'entraînement
    parser.add_argument(
        "--mode",
        type=str,
        default="xor",
        choices=["xor", "text", "image"],
        help="Mode d'entraînement: 'xor' (logique), 'text' (classification de texte), 'image' (classification d'images) (défaut: xor)"
    )

    # Hyperparamètres d'entraînement
    parser.add_argument(
        "--epochs",
        type=int,
        default=50,
        help="Nombre d'époques d'entraînement (défaut: 50)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=64,
        help="Taille des batches (défaut: 64)"
    )
    parser.add_argument(
        "--lr",
        type=float,
        default=1e-3,
        help="Taux d'apprentissage (défaut: 0.001)"
    )

    # Configuration du device
    parser.add_argument(
        "--device",
        type=str,
        default=None,
        help="Device d'entraînement: 'cpu' ou 'cuda' (défaut: auto-détection)"
    )

    # Paramètres de reproductibilité
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Graine aléatoire pour la reproductibilité (défaut: 42)"
    )

    # Options de verbosité
    parser.add_argument(
        "--silent",
        action="store_true",
        help="Mode silencieux (pas de logs détaillés)"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Mode verbeux (logs détaillés, défaut)"
    )

    return parser.parse_args()


def print_banner():
    """Affiche la bannière de démarrage."""
    print("=" * 70)
    print(" " * 15 + "T-RLINKOS TRM++ TRAINING LAUNCHER")
    print("=" * 70)
    print()


def detect_device():
    """Détecte automatiquement le meilleur device disponible."""
    try:
        import torch
        if torch.cuda.is_available():
            device = "cuda"
            print(f"✓ GPU détecté: {torch.cuda.get_device_name(0)}")
            print(f"  Mémoire disponible: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")
        else:
            device = "cpu"
            print("⚠ GPU non disponible, utilisation du CPU")
    except ImportError:
        device = "cpu"
        print("⚠ PyTorch non installé, utilisation du CPU")
    return device


def launch_xor_training(args):
    """Lance l'entraînement sur le dataset XOR.
    
    Args:
        args: Arguments de ligne de commande parsés
    """
    # Import training module
    try:
        from training import train_trlinkos_on_toy_dataset
    except ImportError as e:
        print(f"✗ Erreur: impossible d'importer le module d'entraînement")
        print(f"  Détails: {e}")
        print("  Assurez-vous que PyTorch est installé: pip install torch")
        return 1
    
    print("\n📊 Mode d'entraînement: XOR (Exemple de base)")
    print("-" * 70)
    
    # Déterminer le device
    if args.device is None:
        device = detect_device()
    else:
        device = args.device
        print(f"✓ Device spécifié: {device}")
    
    print()
    print("⚙️ Configuration de l'entraînement:")
    print(f"  • Époques: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Device: {device}")
    print(f"  • Seed: {args.seed}")
    print("-" * 70)
    print()

    # Déterminer la verbosité
    verbose = args.verbose if args.verbose else not args.silent

    # Lancer l'entraînement
    print("🚀 Démarrage de l'entraînement...")
    print()

    try:
        model, history = train_trlinkos_on_toy_dataset(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            verbose=verbose,
        )

        # Afficher le résumé
        print()
        print("=" * 70)
        print(" " * 20 + "📈 RÉSUMÉ DE L'ENTRAÎNEMENT")
        print("=" * 70)
        print()
        print(f"✓ Entraînement terminé avec succès!")
        print()
        print(f"  • Loss finale (train): {history['train_loss'][-1]:.6f}")
        print(f"  • Accuracy finale (train): {history['train_acc'][-1]:.2%}")
        if history['val_loss'] and len(history['val_loss']) > 0:
            print(f"  • Loss finale (validation): {history['val_loss'][-1]:.6f}")
            print(f"  • Accuracy finale (validation): {history['val_acc'][-1]:.2%}")
        print()

        # Déterminer le résultat
        final_acc = history['train_acc'][-1]
        if final_acc >= 0.99:
            print("🎉 Excellent! Le modèle a parfaitement appris le XOR!")
        elif final_acc >= 0.90:
            print("✓ Bon résultat! Le modèle a bien appris le XOR.")
        elif final_acc >= 0.75:
            print("⚠ Résultat moyen. Essayez d'augmenter le nombre d'époques.")
        else:
            print("✗ Résultat insuffisant. Vérifiez les hyperparamètres.")

        print()
        print("=" * 70)
        
        return 0

    except KeyboardInterrupt:
        print()
        print("⚠ Entraînement interrompu par l'utilisateur")
        return 130
    except Exception as e:
        print()
        print(f"✗ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1


def launch_text_training(args):
    """Lance l'entraînement sur un dataset de classification de texte.
    
    Args:
        args: Arguments de ligne de commande parsés
    """
    # Import training module
    try:
        from training import train_trlinkos_on_text_dataset
    except ImportError as e:
        print(f"✗ Erreur: impossible d'importer le module d'entraînement")
        print(f"  Détails: {e}")
        print("  Assurez-vous que PyTorch est installé: pip install torch")
        return 1
    
    print("\n📊 Mode d'entraînement: Classification de TEXTE")
    print("-" * 70)
    
    # Déterminer le device
    if args.device is None:
        device = detect_device()
    else:
        device = args.device
        print(f"✓ Device spécifié: {device}")
    
    print()
    print("⚙️ Configuration de l'entraînement:")
    print(f"  • Dataset: Toy Text Dataset (classification sentiment)")
    print(f"  • Classes: Positif (0) vs Négatif (1)")
    print(f"  • Époques: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Device: {device}")
    print(f"  • Seed: {args.seed}")
    print("-" * 70)
    print()

    # Déterminer la verbosité
    verbose = args.verbose if args.verbose else not args.silent

    # Lancer l'entraînement
    print("🚀 Démarrage de l'entraînement sur texte...")
    print()

    try:
        model, history = train_trlinkos_on_text_dataset(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            verbose=verbose,
        )

        # Afficher le résumé
        print()
        print("=" * 70)
        print(" " * 15 + "📈 RÉSUMÉ DE L'ENTRAÎNEMENT TEXTE")
        print("=" * 70)
        print()
        print(f"✓ Entraînement terminé avec succès!")
        print()
        print(f"  • Loss finale (train): {history['train_loss'][-1]:.6f}")
        print(f"  • Accuracy finale (train): {history['train_acc'][-1]:.2%}")
        if history['val_loss'] and len(history['val_loss']) > 0:
            print(f"  • Loss finale (validation): {history['val_loss'][-1]:.6f}")
            print(f"  • Accuracy finale (validation): {history['val_acc'][-1]:.2%}")
        print()

        # Déterminer le résultat
        final_acc = history['train_acc'][-1]
        if final_acc >= 0.95:
            print("🎉 Excellent! Le modèle classifie très bien les textes!")
        elif final_acc >= 0.85:
            print("✓ Bon résultat! Le modèle a bien appris.")
        elif final_acc >= 0.75:
            print("⚠ Résultat moyen. Essayez d'augmenter le nombre d'époques.")
        else:
            print("✗ Résultat insuffisant. Vérifiez les hyperparamètres.")

        print()
        print("=" * 70)
        
        return 0

    except KeyboardInterrupt:
        print()
        print("⚠ Entraînement interrompu par l'utilisateur")
        return 130
    except Exception as e:
        print()
        print(f"✗ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1


def launch_image_training(args):
    """Lance l'entraînement sur un dataset de classification d'images.
    
    Args:
        args: Arguments de ligne de commande parsés
    """
    # Import training module
    try:
        from training import train_trlinkos_on_image_dataset
    except ImportError as e:
        print(f"✗ Erreur: impossible d'importer le module d'entraînement")
        print(f"  Détails: {e}")
        print("  Assurez-vous que PyTorch est installé: pip install torch")
        return 1
    
    print("\n📊 Mode d'entraînement: Classification d'IMAGES")
    print("-" * 70)
    
    # Déterminer le device
    if args.device is None:
        device = detect_device()
    else:
        device = args.device
        print(f"✓ Device spécifié: {device}")
    
    print()
    print("⚙️ Configuration de l'entraînement:")
    print(f"  • Dataset: Images synthétiques (28x28 RGB)")
    print(f"  • Classes: Clair (0) vs Sombre (1)")
    print(f"  • Époques: {args.epochs}")
    print(f"  • Batch size: {args.batch_size}")
    print(f"  • Learning rate: {args.lr}")
    print(f"  • Device: {device}")
    print(f"  • Seed: {args.seed}")
    print("-" * 70)
    print()

    # Déterminer la verbosité
    verbose = args.verbose if args.verbose else not args.silent

    # Lancer l'entraînement
    print("🚀 Démarrage de l'entraînement sur images...")
    print()

    try:
        model, history = train_trlinkos_on_image_dataset(
            num_epochs=args.epochs,
            batch_size=args.batch_size,
            lr=args.lr,
            device=device,
            seed=args.seed,
            verbose=verbose,
        )

        # Afficher le résumé
        print()
        print("=" * 70)
        print(" " * 15 + "📈 RÉSUMÉ DE L'ENTRAÎNEMENT IMAGE")
        print("=" * 70)
        print()
        print(f"✓ Entraînement terminé avec succès!")
        print()
        print(f"  • Loss finale (train): {history['train_loss'][-1]:.6f}")
        print(f"  • Accuracy finale (train): {history['train_acc'][-1]:.2%}")
        if history['val_loss'] and len(history['val_loss']) > 0:
            print(f"  • Loss finale (validation): {history['val_loss'][-1]:.6f}")
            print(f"  • Accuracy finale (validation): {history['val_acc'][-1]:.2%}")
        print()

        # Déterminer le résultat
        final_acc = history['train_acc'][-1]
        if final_acc >= 0.95:
            print("🎉 Excellent! Le modèle classifie très bien les images!")
        elif final_acc >= 0.85:
            print("✓ Bon résultat! Le modèle a bien appris.")
        elif final_acc >= 0.75:
            print("⚠ Résultat moyen. Essayez d'augmenter le nombre d'époques.")
        else:
            print("✗ Résultat insuffisant. Vérifiez les hyperparamètres.")

        print()
        print("=" * 70)
        
        return 0

    except KeyboardInterrupt:
        print()
        print("⚠ Entraînement interrompu par l'utilisateur")
        return 130
    except Exception as e:
        print()
        print(f"✗ Erreur lors de l'entraînement: {e}")
        import traceback
        traceback.print_exc()
        return 1


def main():
    """Point d'entrée principal."""
    args = parse_args()
    
    # Afficher la bannière
    print_banner()
    
    # Lancer l'entraînement selon le mode
    if args.mode == "xor":
        return launch_xor_training(args)
    elif args.mode == "text":
        return launch_text_training(args)
    elif args.mode == "image":
        return launch_image_training(args)
    else:
        print(f"✗ Mode non supporté: {args.mode}")
        return 1


if __name__ == "__main__":
    sys.exit(main())
