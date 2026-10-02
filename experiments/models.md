# Model
flowchart LR
    P["12 Pokémon<br/>encoders"] --> C["Concatenate<br/>full battle state"]
    F["Field<br/>encoder"] --> C
    H["History<br/>encoder + GRU"] --> C

    C --> T["Trunk<br/>928 → 256 → 128"]

    T --> PH["Policy Head<br/>128 → 10 actions"]
    T --> VH["Value Head<br/>128 → 1"]

# Gen 1 Random Battle

## Reward configs

| name | outcome | own hp | enemy hp | speed | speed scale |
|---|---|---|---|---|---|
| hp-1 | 0 | 0.5 | 0.5 | 0 | 40 |
| outcome-1 | 1 | 0 | 0 | 0 | 40 |
| shapped-1| 0.8 | 0.075 | 0.075 | 0.005 | 40 |
| mixed-1 | 0.5 | 0.25 | 0.25 | 0 | 40 |
| skewed-own-1 | 0.5 | 0.5 | 0 | 0 | 40 |
| skewed-enemy-1 | 0.5 | 0 | 0.5 | 0 | 40 |

## Model configs
### Embedding dimensions: Pokemon
| name | Species | Form | Move | item | ability | status | weather |
|---|---|---|---|---|---|---|---|
| red | 16 | 8 | 16 | 8 | 8 | 8 | 8 |

### Embedding dimensions: History
| name | event type | history ref | history reason |
|---|---|---|---|
| red | 8 | 8 | 8 |

### Model Size
| name | Pkmn hidden | Pkmn output | field hidden | field output |
|---|---|---|---|---|
| red | 128 | 64 | 64 | 32 |

| name | Entry hidden | Entry output | History hidden | Trunk hidden | Trunk output |
|---|---|---|---|---|---|
| red | 96 | 64 | 128 | 256 | 128 |

# Transfomer version
flowchart LR
    P["12 Pokémon<br/>encoders"] --> PP["Project to<br/>Pokémon tokens"]
    F["Field<br/>encoder"] --> FP["Project to<br/>field token"]
    H["History<br/>encoder + GRU"] --> HP["Project to<br/>history token"]
    S["Learned<br/>[STATE] token"] --> X

    PP --> X["Concatenate tokens<br/>[STATE] + 12 Pokémon + field + history"]
    FP --> X
    HP --> X

    X --> TR["Transformer encoder<br/>entity attention"]

    TR --> ST["Final [STATE] token"]

    ST --> PH["Policy Head<br/>128 → 10 actions"]
    ST --> VH["Value Head<br/>128 → 1"]
