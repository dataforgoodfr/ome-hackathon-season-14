# main.py
import argparse
import json
from scoring import default_scorer
from orchestrator import preprocess_text


def read_text_from_file(path: str) -> str:
    with open(path, "r", encoding="utf-8") as f:
        return f.read()


def main():
    parser = argparse.ArgumentParser(description="Theme similarity scorer (embeddings)")
    parser.add_argument("--text", type=str, help="Texte à scorer (entre guillemets).")
    parser.add_argument("--file", type=str, help="Chemin vers un fichier texte à scorer.")
    parser.add_argument("--details", action="store_true", help="Afficher détails (prototypes les plus proches).")
    parser.add_argument("--thresholds", type=str, default=None,
                        help='JSON thresholds, ex: \'{"energy":0.35,"mobility_transport":0.34,"agriculture_alimentation":0.33}\'')

    args = parser.parse_args()

    if not args.text and not args.file:
        parser.error("Il faut fournir --text ou --file")

    text = args.text if args.text else read_text_from_file(args.file)

    split_text = preprocess_text(text)

    # Explain number of duplicate blocs removed
    # Print word count of initial
    # Print word count of final

    scorer = default_scorer()

    for segment in split_text:
        scores, details = scorer.score(segment, return_details=args.details)
    
        # Option: multi-label thresholds
        labels = None
        if args.thresholds:
            thresholds = json.loads(args.thresholds)
            labels = scorer.multilabel(scores, thresholds)
    
        output = {
            "scores": details["scores_by_theme"],
            "best_theme": details["best_theme"],
            "best_score": details["best_score"],
        }
        if labels is not None:
            output["labels"] = labels
        if args.details:
            output["closest_prototypes"] = details.get("closest_prototypes", {})
    
        print(json.dumps(output, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()





