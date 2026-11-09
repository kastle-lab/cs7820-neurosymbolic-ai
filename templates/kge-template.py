
"""
PyKEEN Knowledge Graph Embedding Tutorial
==========================================

Purpose:
    This script provides a template for graduate students to explore
    knowledge graph embedding using the FB15k-237 benchmark dataset,
    which is available directly through PyKEEN.

Learning objectives:
    1. Load the FB15k-237 knowledge graph using PyKEEN.
    2. Explore the structure and characteristics of the dataset.
    3. Understand how entities and relations are represented.
    4. Train a knowledge graph embedding model using TransE.
    5. Experiment with alternative embedding models.
    6. Evaluate model performance using link prediction metrics.
    7. Extract and inspect learned entity and relation embeddings.
    8. Save the trained model and embedding representations.

Requirements:
    pip install pykeen pandas numpy matplotlib

Dataset:
    FB15k-237 is a benchmark knowledge graph derived from Freebase.
    It contains 14,541 entities, 237 relation types, and 310,116
    triples, distributed across predefined training, validation,
    and testing splits.

    PyKEEN automatically downloads and loads the dataset when
    requested for the first time.

Model:
    TransE is used as the default embedding model.

    TransE represents relationships as translations in an embedding
    space:

        h + r approximately equals t

    where h, r, and t represent the head entity, relation, and
    tail entity embeddings, respectively.
"""

# ============================================================
# 1. IMPORT REQUIRED LIBRARIES
# ============================================================

from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from pykeen.datasets import FB15k237
from pykeen.pipeline import pipeline


# ============================================================
# 2. CONFIGURATION
# ============================================================

# Output directory for saving model results and embeddings.

OUTPUT_DIR = Path("pykeen_fb15k237_output")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Select the embedding model.
#
# Students can experiment with alternative models:
#
#   "TransE"
#   "DistMult"
#   "ComplEx"
#   "RotatE"
#   "RESCAL"
#
MODEL_NAME = "TransE"

# Embedding dimensionality.
#
# Each entity and relation will be represented by a vector
# containing EMBEDDING_DIM numerical values.

EMBEDDING_DIM = 100

# Training parameters.

NUM_EPOCHS = 10
BATCH_SIZE = 256
LEARNING_RATE = 0.001

# Reproducibility.

RANDOM_SEED = 42

# Device configuration.
#
# Use "cpu" for compatibility with most computers.
# Change to "cuda" if a compatible NVIDIA GPU is available.

DEVICE = "cpu"


# ============================================================
# 3. LOAD THE FB15K-237 DATASET
# ============================================================

def load_dataset():
    """
    Load FB15k-237 directly from the PyKEEN dataset collection.

    PyKEEN automatically downloads the dataset if it is not
    already available locally.

    FB15k-237 includes three predefined subsets:

        Training:
            Used to learn entity and relation embeddings.

        Validation:
            Used for model selection and hyperparameter tuning.

        Testing:
            Used to evaluate final model performance.

    Unlike manually splitting an RDF graph, this approach
    preserves the benchmark dataset's predefined partitions.

    Returns
    -------
    pykeen.datasets.FB15k237
        Loaded FB15k-237 dataset.
    """

    dataset = FB15k237()

    print("\nFB15k-237 dataset loaded successfully.")

    print("\nDataset characteristics:")
    print(f"Training triples:   {dataset.training.num_triples}")
    print(f"Validation triples: {dataset.validation.num_triples}")
    print(f"Testing triples:    {dataset.testing.num_triples}")

    print(
        f"Number of entities: "
        f"{dataset.training.num_entities}"
    )

    print(
        f"Number of relations: "
        f"{dataset.training.num_relations}"
    )

    return dataset


# ============================================================
# 4. EXPLORE THE DATASET
# ============================================================

def explore_dataset(dataset):
    """
    Explore the structure of the FB15k-237 knowledge graph.

    PyKEEN represents knowledge graph triples internally using
    integer identifiers.

    Each triple follows the structure:

        (head entity ID, relation ID, tail entity ID)

    The TriplesFactory also maintains mappings between the
    original labels and their corresponding integer IDs.

    This function demonstrates how to inspect these mappings
    and retrieve example triples.
    """

    training = dataset.training

    # Retrieve the numerical representation of the triples.

    mapped_triples = training.mapped_triples.cpu().numpy()

    print("\nExample mapped triples:")

    for triple in mapped_triples[:10]:
        print(triple)

    # Retrieve entity and relation mappings.

    entity_to_id = training.entity_to_id
    relation_to_id = training.relation_to_id

    id_to_entity = {
        identifier: label
        for label, identifier in entity_to_id.items()
    }

    id_to_relation = {
        identifier: label
        for label, identifier in relation_to_id.items()
    }

    # Convert the first few triples back into readable labels.

    print("\nExample triples using original labels:")

    for head_id, relation_id, tail_id in mapped_triples[:10]:

        head = id_to_entity[int(head_id)]
        relation = id_to_relation[int(relation_id)]
        tail = id_to_entity[int(tail_id)]

        print(f"({head}, {relation}, {tail})")

    return entity_to_id, relation_to_id


# ============================================================
# 5. TRAIN THE KNOWLEDGE GRAPH EMBEDDING MODEL
# ============================================================

def train_model(dataset):
    """
    Train a knowledge graph embedding model using PyKEEN.

    The pipeline handles:
        - Model initialization.
        - Training.
        - Negative sampling.
        - Validation.
        - Evaluation.
        - Result collection.

    TransE attempts to position entities and relations in
    a vector space such that:

        h + r approximately equals t

    The model learns by distinguishing observed triples
    from artificially generated negative triples.

    Students can experiment with different models and
    hyperparameters to investigate their effects on
    knowledge graph embedding quality.
    """

    print(f"\nStarting model training: {MODEL_NAME}")

    result = pipeline(

        # Use the predefined FB15k-237 partitions.
        training=dataset.training,
        validation=dataset.validation,
        testing=dataset.testing,

        # Select the embedding model.
        model=MODEL_NAME,

        # Model hyperparameters.
        model_kwargs={
            "embedding_dim": EMBEDDING_DIM,
        },

        # Training configuration.
        training_kwargs={
            "num_epochs": NUM_EPOCHS,
            "batch_size": BATCH_SIZE,
        },

        # Optimizer configuration.
        optimizer_kwargs={
            "lr": LEARNING_RATE,
        },

        # Use filtered evaluation.
        #
        # Other known true triples are excluded when
        # calculating the rank of a predicted triple.

        evaluator_kwargs={
            "filtered": True,
        },

        # Reproducibility.
        random_seed=RANDOM_SEED,

        # Computational device.
        device=DEVICE,

    )

    print("\nTraining completed.")

    return result


# ============================================================
# 6. EVALUATE THE MODEL
# ============================================================

def evaluate_model(result):
    """
    Display the model evaluation metrics.

    PyKEEN evaluates link prediction performance by ranking
    candidate entities for a given head entity and relation,
    or for a given relation and tail entity.

    Important evaluation metrics include:

        Mean Rank (MR):
            Average rank of the correct entity.
            Lower values are better.

        Mean Reciprocal Rank (MRR):
            Average reciprocal rank of the correct entity.
            Higher values are better.

        Hits@1:
            Proportion of predictions ranked first.

        Hits@3:
            Proportion of predictions ranked in the top three.

        Hits@10:
            Proportion of predictions ranked in the top ten.

    The evaluation results can be used to compare different
    embedding models and training configurations.
    """

    metrics = result.metric_results.to_dict()

    print("\nModel evaluation results:")

    for metric, value in metrics.items():
        print(f"{metric}: {value}")

    return metrics


# ============================================================
# 7. EXTRACT ENTITY AND RELATION EMBEDDINGS
# ============================================================

def extract_embeddings(result, dataset):
    """
    Extract learned entity and relation embedding vectors.

    The trained PyKEEN model contains separate embedding
    representations for entities and relations.

    Entity embeddings:
        Represent the entities in the knowledge graph.

    Relation embeddings:
        Represent the relationships connecting entities.

    These vectors can be used for:
        - Entity similarity analysis.
        - Clustering.
        - Dimensionality reduction.
        - Knowledge graph visualization.
        - Downstream machine learning applications.

    Returns
    -------
    tuple
        Entity embeddings, relation embeddings, and mappings.
    """

    model = result.model

    # Extract entity embeddings.

    entity_embeddings = (
        model.entity_representations[0]()
        .detach()
        .cpu()
        .numpy()
    )

    # Extract relation embeddings.

    relation_embeddings = (
        model.relation_representations[0]()
        .detach()
        .cpu()
        .numpy()
    )

    # Retrieve original label mappings from the training set.

    entity_to_id = dataset.training.entity_to_id
    relation_to_id = dataset.training.relation_to_id

    print("\nEmbedding information:")

    print(
        f"Entity embedding matrix: "
        f"{entity_embeddings.shape}"
    )

    print(
        f"Relation embedding matrix: "
        f"{relation_embeddings.shape}"
    )

    return (
        entity_embeddings,
        relation_embeddings,
        entity_to_id,
        relation_to_id,
    )


# ============================================================
# 8. SAVE EMBEDDINGS TO CSV
# ============================================================

def save_embeddings(
    entity_embeddings,
    relation_embeddings,
    entity_to_id,
    relation_to_id,
):
    """
    Save the learned embeddings as CSV files.

    Each row represents an entity or relation.
    Each remaining column represents one embedding dimension.

    The original entity and relation labels are included
    to make the resulting files easier to interpret.
    """

    # Reverse the entity and relation mappings.

    id_to_entity = {
        identifier: label
        for label, identifier in entity_to_id.items()
    }

    id_to_relation = {
        identifier: label
        for label, identifier in relation_to_id.items()
    }

    # Create entity embedding DataFrame.

    entity_df = pd.DataFrame(entity_embeddings)

    entity_df.insert(
        0,
        "entity",
        [
            id_to_entity[i]
            for i in range(len(id_to_entity))
        ]
    )

    # Create relation embedding DataFrame.

    relation_df = pd.DataFrame(relation_embeddings)

    relation_df.insert(
        0,
        "relation",
        [
            id_to_relation[i]
            for i in range(len(id_to_relation))
        ]
    )

    # Save CSV files.

    entity_df.to_csv(
        OUTPUT_DIR / "entity_embeddings.csv",
        index=False
    )

    relation_df.to_csv(
        OUTPUT_DIR / "relation_embeddings.csv",
        index=False
    )

    print("\nEmbedding files saved.")

    print(
        OUTPUT_DIR / "entity_embeddings.csv"
    )

    print(
        OUTPUT_DIR / "relation_embeddings.csv"
    )


# ============================================================
# 9. VISUALIZE TRAINING LOSSES
# ============================================================

def plot_training_losses(result):
    """
    Plot the model's training loss across epochs.

    Training loss provides an indication of how the model
    optimizes its learning objective.

    Students should investigate:

        - Does the loss decrease over time?
        - Does the loss stabilize?
        - Does increasing the number of epochs improve results?
        - Does a lower training loss correspond to better
          validation and testing performance?

    A lower training loss does not necessarily imply
    better generalization.
    """

    losses = result.losses

    if losses is None or len(losses) == 0:
        print("No training loss history available.")
        return

    plt.figure(figsize=(8, 5))

    plt.plot(
        range(1, len(losses) + 1),
        losses,
        marker="o"
    )

    plt.xlabel("Epoch")
    plt.ylabel("Training Loss")

    plt.title(
        f"{MODEL_NAME} Training Loss on FB15k-237"
    )

    plt.grid(True)

    plt.tight_layout()

    plt.savefig(
        OUTPUT_DIR / "training_loss.png",
        dpi=300
    )

    plt.show()


# ============================================================
# 10. SAVE THE TRAINED MODEL
# ============================================================

def save_model(result):
    """
    Save the complete trained PyKEEN pipeline.

    Saving the pipeline allows students to reuse the trained
    model without repeating the training process.

    The output directory includes the model and associated
    pipeline information.
    """

    model_path = OUTPUT_DIR / "trained_model"

    result.save_to_directory(model_path)

    print(f"\nModel saved to: {model_path}")


# ============================================================
# 11. MAIN EXECUTION
# ============================================================

def main():

    # Step 1: Load FB15k-237 from PyKEEN.
    dataset = load_dataset()

    # Step 2: Explore the dataset and retrieve mappings.
    entity_to_id, relation_to_id = explore_dataset(dataset)

    # Step 3: Train the selected embedding model.
    result = train_model(dataset)

    # Step 4: Evaluate the model.
    evaluate_model(result)

    # Step 5: Extract learned embeddings.
    (
        entity_embeddings,
        relation_embeddings,
        entity_to_id,
        relation_to_id,
    ) = extract_embeddings(result, dataset)

    # Step 6: Save entity and relation embeddings.
    save_embeddings(
        entity_embeddings,
        relation_embeddings,
        entity_to_id,
        relation_to_id,
    )

    # Step 7: Plot training loss.
    plot_training_losses(result)

    # Step 8: Save the complete trained model.
    save_model(result)

    print("\nFB15k-237 embedding workflow completed.")


# ============================================================
# 12. EXECUTE SCRIPT
# ============================================================

if __name__ == "__main__":
    main()


# ============================================================
# SUGGESTED GRADUATE STUDENT EXPERIMENTS
# ============================================================

"""
Experiment 1: Compare embedding models
---------------------------------------
Train the following models using the same dataset:

    - TransE
    - DistMult
    - ComplEx
    - RotatE

Compare their MRR, Hits@1, Hits@3, and Hits@10 scores.

Question:
    Which model achieves the best link prediction performance?


Experiment 2: Investigate embedding dimensionality
----------------------------------------------------
Change EMBEDDING_DIM to:

    - 50
    - 100
    - 200
    - 300

Question:
    How does embedding dimensionality affect model performance?


Experiment 3: Investigate training epochs
------------------------------------------
Experiment with:

    - 25 epochs
    - 50 epochs
    - 100 epochs
    - 200 epochs

Question:
    Does increasing the number of epochs consistently improve
    link prediction performance?


Experiment 4: Investigate learning rates
------------------------------------------
Experiment with:

    - 0.0001
    - 0.001
    - 0.01

Question:
    How does the learning rate affect convergence and performance?


Experiment 5: Explore entity embeddings
-----------------------------------------
Load entity_embeddings.csv and investigate similarities
between selected entities.

Possible approaches:

    - Cosine similarity.
    - K-means clustering.
    - Principal Component Analysis (PCA).
    - t-SNE visualization.

Question:
    Do semantically related entities have similar embeddings?


Experiment 6: Explore relation embeddings
------------------------------------------
Load relation_embeddings.csv and investigate the learned
representations of different relation types.

Question:
    Can different relation types be distinguished based
    on their embedding vectors?


Experiment 7: Knowledge graph completion
-----------------------------------------
Use the trained model to predict missing links.

For example, given:

    (Entity A, Relation X, ?)

Rank possible tail entities according to their predicted
plausibility.

Question:
    Can the model recover relationships that were withheld
    during training?


Experiment 8: Compare model complexity and performance
--------------------------------------------------------
Compare training time, embedding dimensionality, and
evaluation performance across different models.

Question:
    Is the model with the highest embedding dimension
    necessarily the best model?


Research consideration
----------------------
FB15k-237 is a benchmark dataset with predefined training,
validation, and testing partitions.

The dataset is already mapped into entity and relation IDs
by PyKEEN. Unlike an RDF preprocessing workflow, students
do not need to parse RDF files or manually extract triples.

When interpreting results, consider the limitations of
knowledge graph embeddings, including:

    - Incomplete knowledge graphs.
    - One-to-many and many-to-many relationships.
    - Relation semantics.
    - Negative sampling strategies.
    - Generalization to unseen triples.
    - The distinction between link prediction performance
      and semantic understanding.

Students should report their experimental configurations
and use the same evaluation protocol when comparing models.
"""
