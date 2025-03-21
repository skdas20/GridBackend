from flask import Flask, request, jsonify
import numpy as np
import pickle
import os
from collections import defaultdict
import random
import logging
from flask_cors import CORS

app = Flask(__name__)
CORS(app, resources={r"/api/*": {"origins": ["https://mesh-grid.vercel.app", "http://localhost:3000"]}})

# Configure logging
logging.basicConfig(level=logging.INFO,
                   format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Q-learning parameters
LEARNING_RATE = 0.2
DISCOUNT_FACTOR = 0.9
EXPLORATION_RATE = 0.3
EXPLORATION_DECAY = 0.999
MIN_EXPLORATION_RATE = 0.05

# File to save Q-table
Q_TABLE_FILE = 'q_table.pickle'

# Initialize Q-table as a defaultdict to handle new states automatically
Q_table = defaultdict(lambda: defaultdict(float))

# Try to load existing Q-table if it exists
if os.path.exists(Q_TABLE_FILE):
    try:
        with open(Q_TABLE_FILE, 'rb') as f:
            Q_table_dict = pickle.load(f)
            # Convert the saved dict to a defaultdict
            for state, actions in Q_table_dict.items():
                for action, value in actions.items():
                    Q_table[state][action] = value
        logger.info("Q-table loaded successfully")
    except Exception as e:
        logger.error(f"Error loading Q-table: {e}")
else:
    logger.info("Starting with a new Q-table")

# Current exploration rate
exploration_rate = EXPLORATION_RATE

# State representation functions
def board_to_state(board):
    """Convert board to a string representation for Q-learning"""
    # Sort lines to ensure consistent state representation
    lines_str = ','.join(sorted(key for key, value in board['lines'].items() if value))
    squares_str = ','.join(f"{pos}:{owner}" for pos, owner in board['squares'].items())
    return f"lines:{lines_str}|squares:{squares_str}"

def get_line_from_key(key):
    """Convert line key back to coordinates"""
    parts = key.split('-')
    start = parts[0].split(',')
    end = parts[1].split(',')
    
    return {
        'row1': int(start[0]),
        'col1': int(start[1]),
        'row2': int(end[0]),
        'col2': int(end[1])
    }

# Reward calculation function
def calculate_reward(prev_squares, current_squares, player_id):
    """Calculate reward based on squares completed and strategic positioning"""
    # Count new squares owned by AI
    ai_new_squares = len([sq for sq in current_squares.items() if sq[0] not in prev_squares and sq[1] == player_id])
    
    # Count new squares owned by opponent
    opponent_new_squares = len([sq for sq in current_squares.items() if sq[0] not in prev_squares and sq[1] != player_id])
    
    # Count total squares for both players
    ai_total_squares = len([sq for sq in current_squares.items() if sq[1] == player_id])
    opponent_total_squares = len([sq for sq in current_squares.items() if sq[1] != player_id])
    
    # Base reward for completing squares
    reward = ai_new_squares * 30  # Increased reward for completing squares
    
    # Penalties for giving away squares
    if opponent_new_squares > 0:
        # Higher penalty for giving away squares without gaining any
        if ai_new_squares == 0:
            reward -= 15 * opponent_new_squares
        else:
            # Still a penalty, but reduced if AI also got squares
            reward -= 5 * opponent_new_squares
    
    # Strategic reward: bonus for being ahead
    if ai_total_squares > opponent_total_squares:
        reward += 5 * (ai_total_squares - opponent_total_squares)
    
    # Game progress reward - incentivize early gains
    total_squares_filled = ai_total_squares + opponent_total_squares
    game_progress = total_squares_filled / 16.0  # 0.0 to 1.0
    
    # Early squares are worth more (declining bonus based on game progress)
    if ai_new_squares > 0:
        early_game_bonus = max(0, 10 * (1.0 - game_progress))
        reward += early_game_bonus
    
    # If game is over, calculate final reward
    if total_squares_filled == 16:  # All squares filled
        if ai_total_squares > opponent_total_squares:
            reward += 150  # Increased bonus for winning
        elif ai_total_squares == opponent_total_squares:
            reward += 50   # Bonus for tie
        else:
            reward -= 75   # Penalty for losing
            
        # Larger margin of victory gets higher reward
        victory_margin = abs(ai_total_squares - opponent_total_squares)
        if ai_total_squares > opponent_total_squares:
            reward += victory_margin * 10
    
    return reward

# Save Q-table function
def save_q_table():
    """Save Q-table to disk"""
    try:
        # Convert defaultdict to regular dict for saving
        q_dict = {k: dict(v) for k, v in Q_table.items()}
        with open(Q_TABLE_FILE, 'wb') as f:
            pickle.dump(q_dict, f)
        logger.info("Q-table saved successfully")
    except Exception as e:
        logger.error(f"Error saving Q-table: {e}")

def check_for_completed_squares(board, row1, col1, row2, col2):
    """Check for completed squares after drawing a line"""
    completed_squares = []
    
    # Determine if the line is horizontal or vertical
    is_horizontal = row1 == row2
    
    if is_horizontal:
        # Check square above
        if row1 > 0:
            top = f"{row1-1},{col1}-{row1-1},{col2}"
            left = f"{row1-1},{min(col1, col2)}-{row1},{min(col1, col2)}"
            right = f"{row1-1},{max(col1, col2)}-{row1},{max(col1, col2)}"
            
            if top in board['lines'] and left in board['lines'] and right in board['lines']:
                completed_squares.append(f"{row1-1},{min(col1, col2)}")
        
        # Check square below
        if row1 < 4:
            bottom = f"{row1+1},{col1}-{row1+1},{col2}"
            left = f"{row1},{min(col1, col2)}-{row1+1},{min(col1, col2)}"
            right = f"{row1},{max(col1, col2)}-{row1+1},{max(col1, col2)}"
            
            if bottom in board['lines'] and left in board['lines'] and right in board['lines']:
                completed_squares.append(f"{row1},{min(col1, col2)}")
    else:
        # Check square to the left
        if col1 > 0:
            left = f"{row1},{col1-1}-{row2},{col1-1}"
            top = f"{min(row1, row2)},{col1-1}-{min(row1, row2)},{col1}"
            bottom = f"{max(row1, row2)},{col1-1}-{max(row1, row2)},{col1}"
            
            if left in board['lines'] and top in board['lines'] and bottom in board['lines']:
                completed_squares.append(f"{min(row1, row2)},{col1-1}")
        
        # Check square to the right
        if col1 < 4:
            right = f"{row1},{col1+1}-{row2},{col1+1}"
            top = f"{min(row1, row2)},{col1}-{min(row1, row2)},{col1+1}"
            bottom = f"{max(row1, row2)},{col1}-{max(row1, row2)},{col1+1}"
            
            if right in board['lines'] and top in board['lines'] and bottom in board['lines']:
                completed_squares.append(f"{min(row1, row2)},{col1}")
    
    return completed_squares

def move_completes_square(board, line_key):
    """Check if drawing this line would complete a square"""
    # Parse the line key
    parts = line_key.split('-')
    start = parts[0].split(',')
    end = parts[1].split(',')
    
    row1, col1 = int(start[0]), int(start[1])
    row2, col2 = int(end[0]), int(end[1])
    
    # Create a copy of the board with this line added
    board_copy = {'lines': board['lines'].copy(), 'squares': board['squares'].copy()}
    board_copy['lines'][line_key] = True
    
    # Check if squares are completed
    completed_squares = check_for_completed_squares(board_copy, row1, col1, row2, col2)
    
    return len(completed_squares) > 0

def count_sides_in_box(board, row, col):
    """Count how many sides of a box are already drawn"""
    if row < 0 or row >= 4 or col < 0 or col >= 4:
        return 0  # Invalid box position
        
    top = f"{row},{col}-{row},{col+1}" in board['lines']
    right = f"{row},{col+1}-{row+1},{col+1}" in board['lines']
    bottom = f"{row+1},{col}-{row+1},{col+1}" in board['lines']
    left = f"{row},{col}-{row+1},{col}" in board['lines']
    
    return sum([top, right, bottom, left])

def would_give_away_square(board, line_key):
    """Check if adding this line would set up a square for the opponent to complete"""
    # Parse the line key
    parts = line_key.split('-')
    start = parts[0].split(',')
    end = parts[1].split(',')
    
    row1, col1 = int(start[0]), int(start[1])
    row2, col2 = int(end[0]), int(end[1])
    
    # Create a copy of the board with this line added
    board_copy = {'lines': board['lines'].copy(), 'squares': board['squares'].copy()}
    board_copy['lines'][line_key] = True
    
    # Check if this creates any boxes with 3 sides
    # Horizontal line
    if row1 == row2:
        # Check box above (if it exists)
        if row1 > 0 and count_sides_in_box(board_copy, row1-1, min(col1, col2)) == 3:
            return True
            
        # Check box below (if it exists)
        if row1 < 4 and count_sides_in_box(board_copy, row1, min(col1, col2)) == 3:
            return True
    
    # Vertical line
    else:
        # Check box to the left (if it exists)
        if col1 > 0 and count_sides_in_box(board_copy, min(row1, row2), col1-1) == 3:
            return True
            
        # Check box to the right (if it exists)
        if col1 < 4 and count_sides_in_box(board_copy, min(row1, row2), col1) == 3:
            return True
    
    return False

def find_moves_by_priority(board):
    """Find and classify moves by priority:
    1. Moves that complete squares (highest priority)
    2. Strategic moves that set up chains of boxes
    3. Moves that don't give away squares (medium priority)
    4. All other moves (lowest priority)
    """
    grid_size = 5
    all_moves = []
    
    # Get all available moves
    for row in range(grid_size):
        for col in range(grid_size - 1):
            line_key = f"{row},{col}-{row},{col+1}"
            if line_key not in board['lines']:
                all_moves.append(line_key)
    
    for row in range(grid_size - 1):
        for col in range(grid_size):
            line_key = f"{row},{col}-{row+1},{col}"
            if line_key not in board['lines']:
                all_moves.append(line_key)
    
    # Classify moves
    completing_moves = []
    strategic_moves = []
    safe_moves = []
    unsafe_moves = []
    
    for move in all_moves:
        if move_completes_square(board, move):
            completing_moves.append(move)
        elif would_give_away_square(board, move):
            # Still classify by how many boxes it would give away
            unsafe_moves.append(move)
        else:
            # Check if this move is strategic (could lead to chains)
            if is_strategic_move(board, move):
                strategic_moves.append(move)
            else:
                safe_moves.append(move)
    
    return {
        'completing': completing_moves,
        'strategic': strategic_moves,
        'safe': safe_moves,
        'unsafe': unsafe_moves,
        'all': all_moves
    }

def is_strategic_move(board, line_key):
    """Check if a move is strategic - sets up potential chains of boxes"""
    # Parse the line key
    parts = line_key.split('-')
    start = parts[0].split(',')
    end = parts[1].split(',')
    
    row1, col1 = int(start[0]), int(start[1])
    row2, col2 = int(end[0]), int(end[1])
    
    # Create a copy of the board with this line added
    board_copy = {'lines': board['lines'].copy(), 'squares': board['squares'].copy()}
    board_copy['lines'][line_key] = True
    
    # Check for boxes with exactly 2 sides (developing boxes)
    developing_boxes = []
    
    # Horizontal line
    if row1 == row2:
        # Check boxes related to this line
        if row1 > 0 and count_sides_in_box(board_copy, row1-1, min(col1, col2)) == 2:
            developing_boxes.append((row1-1, min(col1, col2)))
            
        if row1 < 4 and count_sides_in_box(board_copy, row1, min(col1, col2)) == 2:
            developing_boxes.append((row1, min(col1, col2)))
    # Vertical line
    else:
        # Check boxes related to this line
        if col1 > 0 and count_sides_in_box(board_copy, min(row1, row2), col1-1) == 2:
            developing_boxes.append((min(row1, row2), col1-1))
            
        if col1 < 4 and count_sides_in_box(board_copy, min(row1, row2), col1) == 2:
            developing_boxes.append((min(row1, row2), col1))
    
    # If we have at least one developing box, check adjacent boxes
    # to see if we're potentially setting up a chain
    if developing_boxes:
        for box_row, box_col in developing_boxes:
            # Check all adjacent boxes
            adjacent_boxes = [
                (box_row-1, box_col),  # above
                (box_row+1, box_col),  # below
                (box_row, box_col-1),  # left
                (box_row, box_col+1)   # right
            ]
            
            for adj_row, adj_col in adjacent_boxes:
                # Skip invalid boxes
                if adj_row < 0 or adj_row >= 4 or adj_col < 0 or adj_col >= 4:
                    continue
                
                # If an adjacent box has 1 side, this could be part of a chain strategy
                if count_sides_in_box(board_copy, adj_row, adj_col) == 1:
                    return True
        
        # Even without adjacent boxes with 1 side, having multiple developing boxes
        # is generally strategic
        return len(developing_boxes) > 1
        
    return False

def evaluate_risk(board, move):
    """Evaluate how risky a move is based on how many potential squares it gives away"""
    # Parse the line key
    parts = move.split('-')
    start = parts[0].split(',')
    end = parts[1].split(',')
    
    row1, col1 = int(start[0]), int(start[1])
    row2, col2 = int(end[0]), int(end[1])
    
    # Create a copy of the board with this line added
    board_copy = {'lines': board['lines'].copy(), 'squares': board['squares'].copy()}
    board_copy['lines'][move] = True
    
    # Count how many boxes would be at 3 sides
    boxes_at_risk = 0
    
    # Horizontal line
    if row1 == row2:
        # Check box above (if it exists)
        if row1 > 0 and count_sides_in_box(board_copy, row1-1, min(col1, col2)) == 3:
            boxes_at_risk += 1
            
        # Check box below (if it exists)
        if row1 < 4 and count_sides_in_box(board_copy, row1, min(col1, col2)) == 3:
            boxes_at_risk += 1
    
    # Vertical line
    else:
        # Check box to the left (if it exists)
        if col1 > 0 and count_sides_in_box(board_copy, min(row1, row2), col1-1) == 3:
            boxes_at_risk += 1
            
        # Check box to the right (if it exists)
        if col1 < 4 and count_sides_in_box(board_copy, min(row1, row2), col1) == 3:
            boxes_at_risk += 1
    
    return boxes_at_risk

# API endpoint for AI moves
@app.route('/api/move', methods=['POST'])
def get_ai_move():
    global exploration_rate
    
    try:
        # Get data from request
        data = request.get_json()
        board = data['board']
        player_id = data.get('player_id', 'ai-player')
        
        # Get the current state
        current_state = board_to_state(board)
        
        # Get moves classified by priority
        moves_by_priority = find_moves_by_priority(board)
        
        # No available moves
        if not moves_by_priority['all']:
            return jsonify({'move': None})
        
        # Count total completed squares to determine game phase
        total_squares = len(board['squares'])
        game_progress = total_squares / 16.0  # 0.0 to 1.0 indicating progress
        
        # Determine what kind of move to make based on priorities
        chosen_action = None
        
        # Randomly decide whether to explore or exploit
        if random.random() < exploration_rate:
            # Exploration: still prioritize good moves but with randomness
            if moves_by_priority['completing'] and random.random() < 0.95:
                # 95% chance to pick a completing move if available during exploration
                chosen_action = random.choice(moves_by_priority['completing'])
                logger.info(f"Exploring but chose a completing move: {chosen_action}")
            elif moves_by_priority['strategic'] and random.random() < 0.8:
                # 80% chance to pick a strategic move if available during exploration
                chosen_action = random.choice(moves_by_priority['strategic'])
                logger.info(f"Exploring with a strategic move: {chosen_action}")
            elif moves_by_priority['safe'] and random.random() < 0.7:
                # 70% chance to pick a safe move if available during exploration
                chosen_action = random.choice(moves_by_priority['safe'])
                logger.info(f"Exploring with a safe move: {chosen_action}")
            else:
                # Otherwise pick any move
                chosen_action = random.choice(moves_by_priority['all'])
                logger.info(f"Pure exploration with any move: {chosen_action}")
        else:
            # Exploitation: make the best move based on priorities and Q-values
            if moves_by_priority['completing']:
                # If there are moves that complete squares, pick the one with highest Q-value
                chosen_action = max(moves_by_priority['completing'], 
                                  key=lambda move: Q_table[current_state][move])
                logger.info(f"Chose a completing move: {chosen_action}")
            elif moves_by_priority['strategic'] and game_progress < 0.6:
                # Early to mid game: prioritize strategic moves
                chosen_action = max(moves_by_priority['strategic'], 
                                  key=lambda move: Q_table[current_state][move])
                logger.info(f"Chose a strategic move: {chosen_action}")
            elif moves_by_priority['safe']:
                # If there are safe moves, pick the one with highest Q-value
                chosen_action = max(moves_by_priority['safe'], 
                                  key=lambda move: Q_table[current_state][move])
                logger.info(f"Chose a safe move: {chosen_action}")
            elif moves_by_priority['unsafe']:
                # Late game or only unsafe moves remain
                # Sort unsafe moves by risk (fewer boxes at risk is better)
                sorted_unsafe = sorted(moves_by_priority['unsafe'], 
                                       key=lambda move: evaluate_risk(board, move))
                
                # Choose the least risky move
                chosen_action = sorted_unsafe[0] if sorted_unsafe else moves_by_priority['all'][0]
                logger.info(f"Had to choose an unsafe move: {chosen_action} with risk level: {evaluate_risk(board, chosen_action)}")
        
        # Convert chosen action to line coordinates
        line = get_line_from_key(chosen_action)
        
        # Decay exploration rate
        exploration_rate = max(MIN_EXPLORATION_RATE, 
                             exploration_rate * EXPLORATION_DECAY)
        
        # Save the current state and action for updating Q-values later
        with open('current_state.txt', 'w') as f:
            f.write(f"{current_state}|{chosen_action}")
        
        return jsonify({'move': line})
    
    except Exception as e:
        logger.error(f"Error in get_ai_move: {e}")
        return jsonify({'error': str(e)}), 500

# API endpoint to update Q-values after move
@app.route('/api/update', methods=['POST'])
def update_q_values():
    try:
        # Get data from request
        data = request.get_json()
        new_board = data['board']
        reward = data.get('reward', 0)
        completed_squares = data.get('completed_squares', [])
        player_id = data.get('player_id', 'ai-player')
        
        # Try to read previous state and action
        try:
            with open('current_state.txt', 'r') as f:
                state_action = f.read().split('|')
                prev_state = state_action[0]
                action = state_action[1]
        except:
            logger.warning("No previous state found")
            return jsonify({'status': 'no previous state'})
        
        # Convert new board to new state
        new_state = board_to_state(new_board)
        
        # Calculate reward if not provided
        if reward == 0 and completed_squares:
            # Extract previous squares from prev_state
            prev_squares = {}
            if '|squares:' in prev_state:
                squares_part = prev_state.split('|squares:')[1]
                if squares_part:
                    for square_info in squares_part.split(','):
                        if ':' in square_info:
                            pos, owner = square_info.split(':')
                            prev_squares[pos] = owner
            
            reward = calculate_reward(prev_squares, new_board['squares'], player_id)
        
        logger.info(f"Reward for move: {reward}")
        
        # Get current Q-value
        current_q = Q_table[prev_state][action]
        
        # Find maximum Q-value for the new state
        max_future_q = max([Q_table[new_state][a] for a in Q_table[new_state]], default=0)
        
        # Update Q-value using Q-learning formula
        new_q = current_q + LEARNING_RATE * (reward + DISCOUNT_FACTOR * max_future_q - current_q)
        
        # Update Q-table
        Q_table[prev_state][action] = new_q
        
        # Save Q-table periodically (e.g., every 10 updates)
        if random.random() < 0.1:  # 10% chance to save
            save_q_table()
        
        return jsonify({'status': 'updated', 'new_q': new_q})
    
    except Exception as e:
        logger.error(f"Error in update_q_values: {e}")
        return jsonify({'error': str(e)}), 500

# API endpoint to get info about current Q-learning state
@app.route('/api/info', methods=['GET'])
def get_info():
    return jsonify({
        'exploration_rate': exploration_rate,
        'q_table_size': len(Q_table),
        'learning_rate': LEARNING_RATE,
        'discount_factor': DISCOUNT_FACTOR
    })

if __name__ == '__main__':
    # Use PORT environment variable if it exists (for Railway deployment)
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False) 