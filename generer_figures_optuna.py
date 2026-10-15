import argparse
from pathlib import Path
import warnings

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import optuna


def main():
    parser = argparse.ArgumentParser(
        description="Export des graphiques d’une étude Optuna"
    )
    parser.add_argument(
        "--db",
        type=Path,
        default=Path("outputs/optuna_rf/study.db"),
        help="Fichier SQLite Optuna",
    )
    parser.add_argument(
        "--study", default=None, help="Nom de l’étude (automatique si une seule)"
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("outputs/optuna_rf/figures_optuna"),
        help="Dossier de sortie",
    )
    args = parser.parse_args()

    db = args.db.expanduser().resolve()
    if not db.is_file():
        parser.error(f"Base de données introuvable : {db}")
    storage_url = f"sqlite:///{db.as_posix()}"

    summaries = optuna.get_all_study_summaries(storage=storage_url)
    if not summaries:
        parser.error(f"Aucune étude trouvée dans {db}")
    if args.study is None:
        if len(summaries) != 1:
            parser.error(
                "Plusieurs études : préciser --study parmi "
                + ", ".join(s.study_name for s in summaries)
            )
        study_name = summaries[0].study_name
    else:
        study_name = args.study

    study = optuna.load_study(study_name=study_name, storage=storage_url)
    completed = [
        t
        for t in study.trials
        if t.state == optuna.trial.TrialState.COMPLETE and t.value is not None
    ]
    if not completed:
        parser.error(
            "Aucun essai terminé avec une valeur objective : graphiques impossibles."
        )

    args.out.mkdir(parents=True, exist_ok=True)
    print(f"Étude : {study.study_name} | essais terminés : {len(completed)}")
    print(
        f"Meilleur essai : {study.best_trial.number} | score : {study.best_value:.6f}"
    )

    plots = [
        (
            "01_optimization_history",
            "Optimization History Plot",
            optuna.visualization.plot_optimization_history,
            optuna.visualization.matplotlib.plot_optimization_history,
        ),
        (
            "02_parallel_coordinate",
            "Parallel Coordinate Plot",
            optuna.visualization.plot_parallel_coordinate,
            optuna.visualization.matplotlib.plot_parallel_coordinate,
        ),
        (
            "03_hyperparameter_importances",
            "Hyperparameter Importances",
            optuna.visualization.plot_param_importances,
            optuna.visualization.matplotlib.plot_param_importances,
        ),
        (
            "04_slice_plot",
            "Slice Plot",
            optuna.visualization.plot_slice,
            optuna.visualization.matplotlib.plot_slice,
        ),
    ]
    plt.style.use("ggplot")
    for filename, title, plotly_fn, mpl_fn in plots:
        try:
            interactive_fig = plotly_fn(study)
            interactive_fig.update_layout(title=title)
            interactive_fig.write_html(
                str(args.out / f"{filename}.html"), include_plotlyjs=True
            )
            print(f"HTML : {filename}.html")
        except Exception as exc:
            print(f"Impossible de produire le HTML {filename} : {exc}")
        try:
            with warnings.catch_warnings():
                warnings.filterwarnings("ignore", message=".*experimental.*")
                axes = mpl_fn(study)
            if isinstance(axes, matplotlib.axes.Axes):
                fig = axes.figure
            else:
                fig = axes.flat[0].figure
            fig.set_size_inches(15 if filename.endswith("slice_plot") else 12, 7)
            fig.suptitle(title, fontsize=16)
            fig.savefig(args.out / f"{filename}.png", dpi=220, bbox_inches="tight")
            plt.close(fig)
            print(f"PNG  : {filename}.png")
        except Exception as exc:
            print(f"Impossible de produire le PNG {filename} : {exc}")
    print(f"\nFigures enregistrées dans : {args.out.resolve()}")


if __name__ == "__main__":
    main()
