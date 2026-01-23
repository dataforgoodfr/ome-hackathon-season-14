# Start of Season 14 Hackathon

## Data For Good x L’Observatoire des Médias sur l’Écologie

## How to use this repo

This repository is your starting point for the hackathon. Run the different services using

```bash
docker compose up --build inference
```

The `--build` flag is optional in case you have updated the code in order to rebuild the dockerfile.

There are 4 services in the `docker-compose.yml` the most important ones are `inference`, `postgres` and `metabase`. With regards to training models, a little example script is present but feel free to train models on a Google collab environment or on your machine if the training is more optimised (for example when using Apple Silicon). The important service you will be judged on is the `inference` but feel free to store results in the postgres database that has been setup, and to use metabase for data visualisations.

Metabase and data visualisations can help illustrate the value your analysis brings to the solution, or to produce a demo environment, but ultimately you will be judged on the inference solution.

Remember the judging criteria:

- Depth of analysis
- Level of technical maturity
- Frugality
- Use of FOSS tools and models
