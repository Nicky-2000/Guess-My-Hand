import zlib
import math
from itertools import combinations
import tqdm
from tqdm import tqdm
import heapq

def get_rank_from_order(played_order):
    """
    Given the order of 7 played cards, return the rank number (1 to 7!) that corresponds to this order.
    :param played_order: The specific order of the 7 played cards.
    :return: The rank number (1 to 7!) corresponding to the played order. # NEED TO CHANGE THIS TO be 0 - 5040!!!!!!
    """
    available_cards = sorted(played_order)
    rank = 0
    n = len(played_order)

    for i in range(n):
        index = available_cards.index(played_order[i])
        rank += index * math.factorial(n - 1 - i)
        available_cards.pop(index)

    return rank
    
def get_card_order(cards, rank):
    """
    Given a list of 7 cards and a number (rank), return the order in which they should be played.
    :param cards: A list of 7 cards in any order.
    :param rank: The rank number (1 to 7!) that represents the specific permutation.
    :return: A list representing the order in which the cards should be played. (0th index is the first card to play)
    """
    # cards = sorted(cards, key=get_card_value)
    order = []
    # rank -= 1
    n = len(cards)

    for i in range(n, 0, -1):
        factorial = math.factorial(i - 1)
        index = rank // factorial
        order.append(cards.pop(index))
        rank %= factorial

    return order
    
def create_hash_map(cards, index_to_care_about):
    sorted_cards = sorted(cards)
    
    combos = combinations(sorted_cards, 13 - num_cards_to_send)
    # combos = combinations(cards, 13)
    totalCombos = math.comb(len(cards), 13 - num_cards_to_send)
    print("Total Combos: ", totalCombos)
    hash_map = {i: [] for i in range(math.factorial(num_cards_to_send))} 
    
    for combo in tqdm(combos, desc="Hashing combinations", unit="combo", total=totalCombos):
        # sorted_combo = sorted(combo, key=get_card_value)
        hash_value = hash_combination(combo)
        if hash_value == index_to_care_about:
            hash_map[hash_value].append(combo)
    
    return hash_map
    
def hash_combination(cards):
    
    simpleHand = [f"{card}" for card in cards] 
    # return card_hashing.hash_combination_cpp(simpleHand)
    combo_str = ''.join(simpleHand)  
    return zlib.crc32(combo_str.encode()) % (math.factorial(num_cards_to_send))
    
num_cards_to_send = 7
total_cards = list(range(52))
ourHandSorted = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12]
viableCards = [value for value in total_cards if value not in ourHandSorted]

cards_to_hash = ourHandSorted[num_cards_to_send//2:num_cards_to_send//2 + 6] # Get the middle 6 cards
print(cards_to_hash)
# cards_to_hash = ourHandSorted[3:9]
# cards_to_hash = ourHandSorted[num_cards_to_send:] # Get the first 7 cards

ourHandHash = hash_combination(cards_to_hash) # Only hash the last X (6) number of cards
print(f"Player Sending: {ourHandHash}")

cards_to_send = ourHandSorted[:num_cards_to_send//2] + ourHandSorted[num_cards_to_send//2 + 6:]
third_min = cards_to_send[2]
fourth_max = cards_to_send[-4]
print("Cards to Send: ", cards_to_send)

first_7_cards_to_play = get_card_order(cards_to_send, ourHandHash)
print("Cards to play: ", first_7_cards_to_play)

hash_index_to_search = get_rank_from_order(first_7_cards_to_play)
print("Hash Index:", hash_index_to_search)

hash_map = create_hash_map(viableCards, index_to_care_about=hash_index_to_search)

# third_min = cards_to_send[2]
# fourth_max = cards_to_send[-4]
options = []
all_cards_in_options = set()
for combo in hash_map[hash_index_to_search]:
    if (True
        and combo[0] > third_min
        # and combo[-1] < fourth_max
        # and set(combo).issubset(set(cards_to_send)) 
        and not set(cards_to_send).intersection(set(combo))
        # and set(c[num_cards_to_send:]).issubset(set(combo)) # Check if the new cards played are in the combo
        ):
        options.append(combo)
        # all_cards_in_options = set()
        all_cards_in_options.update(combo)

print("Options: ", len(options))
