import random
import numpy as np
from CardGame import Player, Deck
from CardGame import Deck
from tqdm import tqdm
import math
import zlib
from itertools import combinations
# from sklearn.cluster import KMeans

# All Global Maps Used by the players. 
# (Each player only accesses  <Map>[player.name]) so they don't share any data

num_cards_to_send = 7

ourHandHash = {}
first_7_cards_to_play = {}
card_probabilities = {}
hash_index_to_search = {}
hash_map = {}
sorted_first_7_cards_of_team_mate = {}
guesses = {}

def reset_player(player):
    global ourHandHash
    global first_7_cards_to_play
    global card_probabilities
    global hash_map
    global hash_index_to_search
    global sorted_first_7_cards_of_team_mate
    global guesses
    if player.name in ourHandHash:
        del ourHandHash[player.name]
    if player.name in first_7_cards_to_play:
        del first_7_cards_to_play[player.name]
    if player.name in card_probabilities:
        del card_probabilities[player.name]
    if player.name in hash_map:
        del hash_map[player.name]
    if player.name in hash_index_to_search:
        del hash_index_to_search[player.name]
    if player.name in sorted_first_7_cards_of_team_mate:
        del sorted_first_7_cards_of_team_mate[player.name]
    if player.name in guesses:
        del guesses[player.name]

def get_card_value(card):
    rank_map = {
        '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, '10': 10, 
        'J': 11, 'Q': 12, 'K': 13, 'A': 14
    }
    suit_map = {'Hearts': 0.1, 'Diamonds': 0.2, 'Clubs': 0.3, 'Spades': 0.4}
    
    rank = rank_map[card.value]
    suit = suit_map[card.suit]
    
    return rank + suit

def get_card_order(cards, rank):
    """
    Given a list of 7 cards and a number (rank), return the order in which they should be played.
    :param cards: A list of 7 cards in any order.
    :param rank: The rank number (1 to 7!) that represents the specific permutation.
    :return: A list representing the order in which the cards should be played. (0th index is the first card to play)
    """
    cards = sorted(cards, key=get_card_value)
    order = []
    # rank -= 1  
    n = len(cards)

    for i in range(n, 0, -1):
        factorial = math.factorial(i - 1)
        index = rank // factorial
        order.append(cards.pop(index))
        rank %= factorial

    return order # NEED TO CHANGE THIS TO be 0 - 5040!!!!!!

def get_rank_from_order(played_order):
    """
    Given the order of 7 played cards, return the rank number (1 to 7!) that corresponds to this order.
    :param played_order: The specific order of the 7 played cards.
    :return: The rank number (1 to 7!) corresponding to the played order. # NEED TO CHANGE THIS TO be 0 - 5040!!!!!!
    """
    available_cards = sorted(played_order, key=get_card_value)
    rank = 0
    n = len(played_order)

    for i in range(n):
        index = available_cards.index(played_order[i])
        rank += index * math.factorial(n - 1 - i)
        available_cards.pop(index)

    return rank  # Weird hack Need to look at the ranks of the encoding and decoding. I think we might be off by 1

def hash_combination(cards):
    simpleHand = [f"{card.value}{card.suit[0]}" for card in cards] 
    # return card_hashing.hash_combination_cpp(simpleHand)
    combo_str = ''.join(simpleHand)  
    return zlib.crc32(combo_str.encode()) % (math.factorial(num_cards_to_send))

def create_hash_map(cards, index_to_care_about):
    sorted_cards = sorted(cards, key=get_card_value)
    
    combos = combinations(sorted_cards, 13 - num_cards_to_send)
    # combos = combinations(cards, 13)
    totalCombos = math.comb(len(cards), 13 - num_cards_to_send)
    print("Total Combos: ", totalCombos)
    hash_map = {i: [] for i in range(math.factorial(num_cards_to_send))} 
    
    for combo in combos:
        # sorted_combo = sorted(combo, key=get_card_value)
        hash_value = hash_combination(combo)
        if hash_value == index_to_care_about:
            hash_map[hash_value].append(combo)
    
    return hash_map


# three_min_four_max = ourHandSorted[0:3] + ourHandSorted[-4:]
# print("three_min_four_max: ", three_min_four_max)
# cards_to_hash = list(set(ourHandSorted) - set(three_min_four_max))
# ourHandHash[player.name] = hash_combination(cards_to_hash) # Only hash the cards we don't play
# print(f"Player: {player.name} Sending: {ourHandHash[player.name]}")
# first_7_cards_to_play[player.name] = get_card_order(three_min_four_max, ourHandHash[player.name]) # Should prob use sorted hand here

def playing(player: Player, deck: Deck):
    global ourHandHash
    global first_7_cards_to_play
    turn = len(player.played_cards) + 1
    if turn == 1:
        print("NEW ROUND!!!")
        reset_player(player)
        # Hash our cards and figure out what we are going to play
        # simpleHand = [f"{card.value}{card.suit[0]}" for card in player.hand]
        ourHandSorted = sorted(player.hand, key=get_card_value)

        three_min_four_max = ourHandSorted[0:3] + ourHandSorted[-4:]
        print("three_min_four_max: ", three_min_four_max)
        cards_to_hash = sorted((set(ourHandSorted) - set(three_min_four_max)),key=get_card_value)
        ourHandHash[player.name] = hash_combination(cards_to_hash) # Only hash the cards we don't play

        print(f"Player: {player.name} Sending: {ourHandHash[player.name]}")
        first_7_cards_to_play[player.name] = get_card_order(three_min_four_max, ourHandHash[player.name]) # Should prob use sorted hand here
        
        # ourHandSorted = sorted(player.hand, key=get_card_value)
        # ourHandHash[player.name] = hash_combination(ourHandSorted[num_cards_to_send:]) # Only hash the last X number of cards
        # print(f"Player: {player.name} Sending: {ourHandHash[player.name]}")

        # first_7_cards_to_play[player.name] = get_card_order(ourHandSorted[:num_cards_to_send], ourHandHash[player.name]) # Should prob use sorted hand here
    if turn <= num_cards_to_send:
        if first_7_cards_to_play[player.name]:
            card_to_play = first_7_cards_to_play[player.name].pop(0) # get first element in list
            return player.hand.index(card_to_play)
        else:
            raise ValueError("first_7_cards_to_play was not initialized properly.")
        # card_to_play = first_7_cards_to_play.pop(0)
        # return player.hand.index(card_to_play)
    else: 
        return random.randint(0, len(player.hand) - 1)

def guessing(player, cards, round):
    global card_probabilities
    global hash_map
    global hash_index_to_search
    global sorted_first_7_cards_of_team_mate
    global guesses

    if round == 1:
        reset_player(player)
    
    teamMatesPlayedCards = get_team_mates_exposed_cards(player)
    
    if round == num_cards_to_send: # only create map on round 7
        cards_copy = cards.copy()
        # viableCards = list(set(get_viable_cards(cards_copy, player)) - set(teamMatesPlayedCards))
        viableCards = list(set(get_viable_cards(cards_copy, player)) - set(teamMatesPlayedCards))
        hash_index_to_search[player.name] = get_rank_from_order(teamMatesPlayedCards)
        print(f"Player: {player.name} Received: {hash_index_to_search[player.name]}")

        hash_map[player.name] = create_hash_map(viableCards, index_to_care_about=hash_index_to_search[player.name])
        sorted_first_7_cards_of_team_mate[player.name] = get_tuple_representation_of_cards(sorted(teamMatesPlayedCards, key=get_card_value))
        # hash_index_to_search[player.name] = get_rank_from_order(teamMatesPlayedCards)

    if player.name in hash_index_to_search:
        personal_hash_map = hash_map[player.name]
        personal_hash_index = hash_index_to_search[player.name]
        
        # max_team_mate_played_card = max(get_card_value(card) for card in teamMatesPlayedCards[0:num_cards_to_send])
        teamMatesFirst7 = teamMatesPlayedCards[0:num_cards_to_send].copy()
        teamMatesFirst7 = sorted(teamMatesFirst7, key=get_card_value)
        min_max_team_mate_played_card = teamMatesFirst7[-4]
        max_min_team_mate_played_card = teamMatesFirst7[2]

        options = []
        all_cards_in_options = set()
        for combo in personal_hash_map[personal_hash_index]:
            if round > num_cards_to_send:
                if (get_card_value(combo[0]) > get_card_value(max_min_team_mate_played_card )
                    and get_card_value(combo[-1]) < get_card_value(min_max_team_mate_played_card)
                    and set(combo).issubset(set(cards)) 
                    and not set(get_other_teams_exposed_cards(player)).intersection(set(combo))
                    and set(teamMatesPlayedCards[num_cards_to_send:]).issubset(set(combo)) # Check if the new cards played are in the combo
                    ):
                    options.append(combo)
                    all_cards_in_options.update(combo)
            else: 
                if (get_card_value(combo[0]) > get_card_value(max_min_team_mate_played_card)
                    and get_card_value(combo[-1]) < get_card_value(min_max_team_mate_played_card)
                    and set(combo).issubset(set(cards)) 
                    and not set(get_other_teams_exposed_cards(player)).intersection(set(combo))
                    ):
                    options.append(combo)
                    all_cards_in_options.update(combo)
        
        personal_hash_map[personal_hash_index] = options

        if len(options) > 1:
            print(f"Number of Options: {len(options)} on round {round}")
            print(f"Number of Cards in Options: {len(all_cards_in_options)} on round {round}")
            non_viable_cards = set(cards) - all_cards_in_options
            for card in non_viable_cards:
                if card in card_probabilities[player.name]:
                    del card_probabilities[player.name][card]
            update_card_probs(cards, card_probabilities, player, round)
            
            card_probs = card_probabilities[player.name].copy()
            guess = sorted(card_probs, key=card_probs.get, reverse=True)[:13 - round]
            guesses[player.name].append(guess)
            return guess
        
        chosen = random.choice(options)
        return list(set(chosen) - set(teamMatesPlayedCards))
    else:
        if round == 1:
            init_card_probs(cards, card_probabilities, player)
            guesses[player.name] = []

        update_card_probs(cards, card_probabilities, player, round)
        card_probs = card_probabilities[player.name].copy()
        guess = sorted(card_probs, key=card_probs.get, reverse=True)[:13 - round]
        guesses[player.name].append(guess)
        return guess

def init_card_probs(cards, card_probabilities, player):
    # Set up dict
    card_probabilities[player.name] = {}
    for card in cards:
        card_probabilities[player.name][card] = 1


def update_card_probs(cards, card_probabilities, player, round):
    # remove the cards that were played
    for card in set(cards) - set(get_viable_cards(cards, player)):
        if card in card_probabilities[player.name]:
            del card_probabilities[player.name][card]

    # remove partner's card
    partner_card = get_team_mates_exposed_cards(player)[-1]
    if partner_card in card_probabilities[player.name]:
        del card_probabilities[player.name][partner_card]

    # first round only
    if len(guesses[player.name]) == 0:
        return
    
    team_mates_cards_set = set(get_team_mates_exposed_cards(player))
    other_teams_exposed_cards_set = set(get_other_teams_exposed_cards(player))

    combined_probs_from_guesses = {key: 1 for key in card_probabilities[player.name]}
    cards_with_non_zero_probability = set(card_probabilities[player.name].keys())
    for guess_index, guess in enumerate(guesses[player.name]):
        # print(guess_index)
        # print(player.cVals)
        cVal_for_guess = player.cVals[guess_index]
        guess_set = set(guess)

        prob_numerator_for_cards_in_guess = cVal_for_guess - len(guess_set.intersection(team_mates_cards_set))
        prob_denominator_for_cards_in_guess = len(guess_set.intersection(cards_with_non_zero_probability))

        viable_cards_not_in_guess = set(cards_with_non_zero_probability) - set(guess_set)
        num_viable_cards_not_in_guess = len(viable_cards_not_in_guess)

        num_cards_not_in_guess_team_mate_has_played = len(team_mates_cards_set) - len(guess_set.intersection(team_mates_cards_set))
        
        prob_numerator_for_cards_not_in_guess = len(guess) - cVal_for_guess - num_cards_not_in_guess_team_mate_has_played
        prob_denominator_for_cards_not_in_guess = num_viable_cards_not_in_guess


        for card in card_probabilities[player.name]:
            if card in guess:
                combined_probs_from_guesses[card] *= (prob_numerator_for_cards_in_guess / prob_denominator_for_cards_in_guess) if prob_denominator_for_cards_in_guess > 0 else 0
            else:
                combined_probs_from_guesses[card] *= (prob_numerator_for_cards_not_in_guess / prob_denominator_for_cards_not_in_guess) if prob_denominator_for_cards_not_in_guess > 0 else 0
    # Now combine all the probabilities from each guess

    # max_team_mate_played_card = max(get_card_value(card) for card in team_mates_cards_set)
    
    # for card in combined_probs_from_guesses:
        # if get_card_value(card) > max_team_mate_played_card:
            # pass
            # combined_probs_from_guesses[card] *= 100

    for card in set(cards) - set(get_viable_cards(cards, player)):
        if card in combined_probs_from_guesses and combined_probs_from_guesses[card] == 0:
            del combined_probs_from_guesses[card]

    if round == 1:
        teamMatesPlayedCardVal = get_card_value(get_team_mates_exposed_cards(player)[0])
        # print(f"teamMatesPlayedCardVal: {teamMatesPlayedCardVal}")
        # Calculate the lowest and highest possible card values
        lowest_possible_value = 2 + 0.1  # Assuming 2 of Hearts is the lowest
        highest_possible_value = 14 + 0.4  # Assuming Ace of Spades is the highest

        # Calculate the distance from the lowest and highest possible values
        distance_from_lowest = abs(teamMatesPlayedCardVal - lowest_possible_value)
        distance_from_highest = abs(teamMatesPlayedCardVal - highest_possible_value)

        # Assign teamMatesPlayedCard to min or max based on distance
        if distance_from_lowest < distance_from_highest:
            # Assign this card as min
            min1 = teamMatesPlayedCardVal

            # Boost the probabilities of cards that above min
            for card in set(cards) - set(get_viable_cards(cards, player)):
                if card in combined_probs_from_guesses:
                    if (get_card_value(card) > min1):
                        combined_probs_from_guesses[card] *= 1.5
        else:
            #Assign this card as max
            max1= teamMatesPlayedCardVal

            # Boost the probabilities of cards that below max
            for card in set(cards) - set(get_viable_cards(cards, player)):
                if card in combined_probs_from_guesses:
                    if (get_card_value(card) > max1):
                        combined_probs_from_guesses[card] *= 1.5

    if round == 2:
        teamMatesPlayedCards = get_team_mates_exposed_cards(player)

        # Get the card values for the two played cards
        card1_value = float(get_card_value(teamMatesPlayedCards[0]))
        card2_value = float(get_card_value(teamMatesPlayedCards[1]))

        # Assign min and max based on the card values
        min_value = min(card1_value, card2_value)
        max_value = max(card1_value, card2_value)

        # Boost the probabilities of cards between min and max
        for card in set(cards) - set(get_viable_cards(cards, player)):
            if card in combined_probs_from_guesses:
                card_value = get_card_value(card)
                if min_value < card_value < max_value:
                    combined_probs_from_guesses[card] *= 1.2

    # Example usage for your card game in round 3
    if round == 3 or round == 4 or  round == 5 or round == 6:   
        teamMatesPlayedCards = get_team_mates_exposed_cards(player)
        card_values = [get_card_value(card) for card in teamMatesPlayedCards]
        
        # Ensure the card values are unique
        card_values = list(set(card_values))
        
        # Run the KMeans algorithm
        clusters = kmeans(card_values)
        
        # Assign elements to min and max arrays based on the cluster means
        min_array, max_array = assign_min_max_clusters(clusters)

        max_min_value = max(min_array)
        min_max_value = min(max_array)

        # Boost probabilities of cards between max_min_value and min_max_value
        for card in set(cards) - set(get_viable_cards(cards, player)):
            card_value = get_card_value(card)
            if max_min_value < card_value < min_max_value:
                if card in combined_probs_from_guesses:
                    combined_probs_from_guesses[card] *= 1.5  # Boost by 10%

        if round > 5 and len(min_array) == 3:
            min_min_value = min(min_array)
            for card in set(cards) - set(get_viable_cards(cards, player)):
                card_value = get_card_value(card)
                if card_value < min_min_value:
                    if card in combined_probs_from_guesses:
                        combined_probs_from_guesses[card] *= 0

        if round > 5 and len(max_array) == 4:
            max_max_value = max(max_array)
            for card in set(cards) - set(get_viable_cards(cards, player)):
                card_value = get_card_value(card)
                if card_value > max_max_value:
                    if card in combined_probs_from_guesses:
                        combined_probs_from_guesses[card] *= 0 
        # return combined_probs_from_guesses
        


    
    if round > 7: 
        teamMatesPlayedCards = get_team_mates_exposed_cards(player)
        teamMatesFirst7 = teamMatesPlayedCards[0:num_cards_to_send].copy()
        teamMatesFirst7 = sorted(teamMatesFirst7, key=get_card_value)
        min_max_team_mate_played_card = teamMatesFirst7[-4]
        max_min_team_mate_played_card = teamMatesFirst7[2]

        for card in set(cards) - set(get_viable_cards(cards, player)):
            if card in combined_probs_from_guesses:
                if (get_card_value(card) > get_card_value(max_min_team_mate_played_card) 
                    and get_card_value(card) < get_card_value(min_max_team_mate_played_card)
                    ):
                    combined_probs_from_guesses[card] *= 1.5

                

    print(len(combined_probs_from_guesses))
            
    card_probabilities[player.name] = combined_probs_from_guesses



    # # remove cards in last guess that were played
    # last_guess = []
    # for card in guesses[player.name][-1]:
    #     if card in card_probabilities[player.name]:
    #         last_guess.append(card)

    # # count how many correct guesses 'remain' in that guess
    # played_guessed_right = set(guesses[player.name][-1]).intersection(set(get_team_mates_exposed_cards(player)))
    # last_cval = player.cVals[-1] - len(played_guessed_right)

    # # update probabilities of cards in last guess
    # for card in last_guess:
    #     if card in card_probabilities[player.name]:
    #         card_probabilities[player.name][card] = last_cval / len(last_guess)

    # # update probabilities of cards not in last guess
    # for card in set(cards) - set(last_guess):
    #     if card in card_probabilities[player.name]:
    #         card_probabilities[player.name][card] = (len(guesses[player.name][-1]) - 1 - last_cval) / (len(card_probabilities[player.name]) - len(last_guess))

    # # update probabilities of remaining valid cards
    # for card in get_viable_cards(cards, player):
    #     if card in card_probabilities[player.name]:
    #         card_probabilities[player.name][card] = 1 / len(card_probabilities[player.name])


# probability_of_guessed_cards = Numerator_guessed/ Denominator_guessed

# Numerator_guessed = cvalue_for_that_turn - number_of_cards_in_guess_team_mate_has_played

# Denominator_guessed = "Num viable cards not in guess?" 

# # length_of_the_guess - number_of_cards_in_guess_team_mate_has_played - number_of_cards_in_guess_that_were_played_by_other_team


# Probability_of_non_guessed_cards = Numerator_Not_guessed/ Denominator_not_guessed

# Numerator_Not_guessed = (MaxCValueForTurn - cvalue_for_that_turn) - number_of_cards_not_in_guess_team_mate_has_played
# Denominator_not_guessed = Num_viable_cards_not_in_guess - length_of_the_guess - number_of_cards_in_guess_team_mate_has_played - number_of_cards_in_guess_that_were_played_by_other_team

# Probability_card_in_previous_guess = 




def get_tuple_representation_of_cards(cards):
    return [(card.value, card.suit) for card in cards]
def get_team_mates_exposed_cards(player) -> list:
    if player.name == "East":
        return player.exposed_cards["West"]
    elif player.name == "West":
        return player.exposed_cards["East"]
    elif player.name == "North":
        return player.exposed_cards["South"]
    elif player.name == "South":
        return player.exposed_cards["North"]

def get_viable_cards(cards, player):
    card_set = set(cards)
    player_hand_set = set(player.hand).union(set(player.played_cards))
    other_teams_exposed_cards = set(get_other_teams_exposed_cards(player))
    viable_cards = card_set - player_hand_set - other_teams_exposed_cards
    return list(viable_cards)


def get_other_teams_exposed_cards(player) -> list:
    if player.name == "East" or player.name == "West":
        return player.exposed_cards["North"] + player.exposed_cards["South"]
    elif player.name == "North" or player.name == "South":
        return player.exposed_cards["East"] + player.exposed_cards["West"]
    
def initialize_centroids(card_values, k):
    # Randomly choose k unique card values as initial centroids
    return random.sample(card_values, k)

def assign_clusters(card_values, centroids):
    clusters = {}
    for card_value in card_values:
        # Calculate distance from each centroid and assign card_value to the closest one
        distances = [abs(card_value - centroid) for centroid in centroids]
        cluster_index = np.argmin(distances)
        if cluster_index not in clusters:
            clusters[cluster_index] = []
        clusters[cluster_index].append(card_value)
    return clusters

def update_centroids(clusters):
    # Update centroids by calculating the mean of the current clusters
    return [np.mean(clusters[cluster]) for cluster in clusters]

def kmeans(card_values, k=2, max_iters=100):
    centroids = initialize_centroids(card_values, k)
    for _ in range(max_iters):
        clusters = assign_clusters(card_values, centroids)
        new_centroids = update_centroids(clusters)
        # Stop if centroids don't change
        if np.allclose(centroids, new_centroids):
            break
        centroids = new_centroids
    return clusters

def assign_min_max_clusters(clusters):
    # Calculate the mean of each cluster
    cluster_means = {i: np.mean(clusters[i]) for i in clusters}
    # Sort clusters by their means
    sorted_clusters = sorted(cluster_means, key=cluster_means.get)

    # Ensure the first is the min cluster and the second is the max cluster
    min_array = clusters[sorted_clusters[0]]
    max_array = clusters[sorted_clusters[1]]
    return min_array, max_array