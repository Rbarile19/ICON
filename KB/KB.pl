% Regola 1: Identificare se un paziente è a rischio di attacco cardiaco
paziente_a_rischio(Eta, Colesterolo, AnginaEsercizio, TipoDolore) :-
    Eta > 50,
    Colesterolo > 240,
    AnginaEsercizio = 1,
    TipoDolore > 1.

% Regola 2: Tipo di attacco basato su pressione e picco ST
tipo_attacco(Eta, Colesterolo, AnginaEsercizio, TipoDolore, Pressione, PiccoPrecedente, alto) :-
    paziente_a_rischio(Eta, Colesterolo, AnginaEsercizio, TipoDolore),
    Pressione > 160,
    PiccoPrecedente > 2.5.

tipo_attacco(Eta, Colesterolo, AnginaEsercizio, TipoDolore, Pressione, PiccoPrecedente, medio) :-
    paziente_a_rischio(Eta, Colesterolo, AnginaEsercizio, TipoDolore),
    Pressione > 140, Pressione =< 160,
    PiccoPrecedente =< 2.5.

tipo_attacco(Eta, Colesterolo, AnginaEsercizio, TipoDolore, Pressione, PiccoPrecedente, basso) :-
    paziente_a_rischio(Eta, Colesterolo, AnginaEsercizio, TipoDolore),
    Pressione =< 140.

% Regola 3: Età media dei pazienti
eta_media(ListaEta, EtaMedia) :-
    sumlist(ListaEta, Somma),
    length(ListaEta, NumeroPazienti),
    EtaMedia is Somma / NumeroPazienti.

% Regola 4: Determinare se un paziente può avere un attacco cardiaco
puo_avere_attacco_cardiaco_prob(Eta, TipoDolore, AnginaEsercizio, Pendenza, NumeroVasi, RisultatoThallium, FrequenzaCardiaca, PiccoPrecedente, PuoAvereAttacco) :-
    (
        (TipoDolore == 0 -> ValoreCondizione1 = 0; ValoreCondizione1 = 1),
        (AnginaEsercizio == 1 -> ValoreCondizione2 = 0; ValoreCondizione2 = 1),
        ((Pendenza == 0; Pendenza == 1) -> ValoreCondizione3 = 0; ValoreCondizione3 = 1),
        (NumeroVasi =< 3 -> ValoreCondizione4 = 0; ValoreCondizione4 = 1),
        (RisultatoThallium == 3 -> ValoreCondizione5 = 0; ValoreCondizione5 = 1),
        ((Eta > 50; Eta < 30) -> ValoreCondizione6 = 0; ValoreCondizione6 = 1),
        (FrequenzaCardiaca < 130 -> ValoreCondizione7 = 0; ValoreCondizione7 = 1),
        (PiccoPrecedente > 2.0 -> ValoreCondizione8 = 0; ValoreCondizione8 = 1)
    ),
    SommaCondizioni is ValoreCondizione1 + ValoreCondizione2 + ValoreCondizione3 + ValoreCondizione4 + ValoreCondizione5 + ValoreCondizione6 + ValoreCondizione7 + ValoreCondizione8,
    (SommaCondizioni > 4 -> PuoAvereAttacco = si; PuoAvereAttacco = no).

% Regola 5: Condizione cardiovascolare grave
condizione_grave(Colesterolo, NumeroVasi, FrequenzaCardiaca) :-
    Colesterolo > 300,
    NumeroVasi >= 2,
    FrequenzaCardiaca < 100.
