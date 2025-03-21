const express = require('express');
const http = require('http');
const socketIo = require('socket.io');
const { exec } = require('child_process');
const axios = require('axios');
const cors = require('cors');

const app = express();
const server = http.createServer(app);
const io = socketIo(server, {
    cors: {
        origin: ["https://mesh-grid.vercel.app", "http://localhost:3000"],
        methods: ["GET", "POST"],
        credentials: true
    }
});

// Enable CORS for REST endpoints
app.use(cors({
    origin: ["https://mesh-grid.vercel.app", "http://localhost:3000"],
    methods: ["GET", "POST"],
    credentials: true
}));

// Serve static files
app.use(express.static('public'));

// Game rooms storage
const gameRooms = {};

// AI server process
let aiServer = null;

// Start the AI Flask server
function startAIServer() {
    if (!aiServer) {
        aiServer = exec('python ai_server.py', (error, stdout, stderr) => {
            if (error) {
                console.error(`AI server error: ${error}`);
                return;
            }
            console.log(`AI server output: ${stdout}`);
        });
        
        console.log('AI server started');
    }
}

// Handle client connection
io.on('connection', (socket) => {
    console.log('A user connected:', socket.id);
    
    // Create a new game room
    socket.on('createGame', () => {
        const roomCode = generateRoomCode();
        gameRooms[roomCode] = {
            players: [socket.id],
            board: createEmptyBoard(),
            currentPlayer: socket.id,
            scores: {},
            aiMode: false
        };
        
        gameRooms[roomCode].scores[socket.id] = 0;
        
        socket.join(roomCode);
        socket.emit('gameCreated', { roomCode });
        console.log(`Game created with code: ${roomCode}`);
    });
    
    // Join an existing game room
    socket.on('joinGame', ({ roomCode }) => {
        if (gameRooms[roomCode] && gameRooms[roomCode].players.length < 2) {
            gameRooms[roomCode].players.push(socket.id);
            gameRooms[roomCode].scores[socket.id] = 0;
            
            socket.join(roomCode);
            socket.emit('gameJoined', { roomCode });
            
            // Notify the first player that someone joined
            io.to(gameRooms[roomCode].players[0]).emit('playerJoined');
            
            console.log(`Player ${socket.id} joined room: ${roomCode}`);
        } else {
            socket.emit('joinError', { message: 'Room not found or full' });
        }
    });
    
    // Create AI game
    socket.on('createAIGame', () => {
        const roomCode = generateRoomCode();
        gameRooms[roomCode] = {
            players: [socket.id, 'ai-player'],
            board: createEmptyBoard(),
            currentPlayer: socket.id,
            scores: {},
            aiMode: true
        };
        
        // Initialize scores
        gameRooms[roomCode].scores[socket.id] = 0;
        gameRooms[roomCode].scores['ai-player'] = 0;
        
        socket.join(roomCode);
        socket.emit('gameCreated', { roomCode, aiMode: true });
        
        // Start the AI server
        startAIServer();
        
        console.log(`AI game created with code: ${roomCode}`);
    });
    
    // Handle player moves
    socket.on('move', ({ roomCode, line }) => {
        const room = gameRooms[roomCode];
        if (!room) return;
        
        // Only allow moves from the current player
        if (room.currentPlayer !== socket.id) {
            socket.emit('notYourTurn');
            return;
        }
        
        // Update the board with the new line
        const { row1, col1, row2, col2 } = line;
        const lineKey = getLineKey(row1, col1, row2, col2);
        
        // Check if line already exists
        if (room.board.lines[lineKey]) {
            socket.emit('invalidMove');
            return;
        }
        
        // Add line to the board
        room.board.lines[lineKey] = true;
        
        // Check if a square was completed
        const completedSquares = checkForCompletedSquares(room.board, row1, col1, row2, col2);
        let extraTurn = false;
        
        if (completedSquares.length > 0) {
            // Award points to the current player
            room.scores[socket.id] += completedSquares.length;
            
            // Mark squares as owned
            completedSquares.forEach(square => {
                room.board.squares[square] = socket.id;
            });
            
            extraTurn = true;
        }
        
        // Broadcast the move to all players in the room
        io.to(roomCode).emit('moveMade', {
            line,
            player: socket.id,
            completedSquares,
            scores: room.scores
        });
        
        // Check if game is over
        if (isGameOver(room.board)) {
            io.to(roomCode).emit('gameOver', { scores: room.scores });
            return;
        }
        
        // Change turn if no square was completed
        if (!extraTurn) {
            const nextPlayerIndex = room.players.indexOf(socket.id) === 0 ? 1 : 0;
            room.currentPlayer = room.players[nextPlayerIndex];
            
            // If AI's turn, make AI move
            if (room.aiMode && room.currentPlayer === 'ai-player') {
                makeAIMove(roomCode);
            }
        }
        
        // Notify players about the current turn and scores
        io.to(roomCode).emit('turnChange', { 
            currentPlayer: room.currentPlayer,
            scores: room.scores  // Include scores with the turn change
        });
    });
    
    // Handle disconnection
    socket.on('disconnect', () => {
        console.log('User disconnected:', socket.id);
        
        // Find and clean up any rooms the player was in
        Object.keys(gameRooms).forEach(roomCode => {
            const room = gameRooms[roomCode];
            const playerIndex = room.players.indexOf(socket.id);
            
            if (playerIndex !== -1) {
                // Notify other player about disconnection
                if (room.players.length > 1 && !room.aiMode) {
                    const otherPlayerIndex = playerIndex === 0 ? 1 : 0;
                    const otherPlayer = room.players[otherPlayerIndex];
                    io.to(otherPlayer).emit('playerDisconnected');
                }
                
                // Remove the room if it's a regular game
                if (!room.aiMode) {
                    delete gameRooms[roomCode];
                }
            }
        });
    });
});

// Make AI move
function makeAIMove(roomCode) {
    const room = gameRooms[roomCode];
    if (!room || !room.aiMode) return;
    
    // Use RL-based AI server for decisions
    
    // Add a small delay to simulate thinking
    setTimeout(async () => {
        try {
            // Step 1: Ask AI server for the best move
            const aiResponse = await axios.post('http://localhost:5000/api/move', {
                board: room.board,
                player_id: 'ai-player'
            });
            
            const aiLine = aiResponse.data.move;
            
            if (!aiLine) {
                console.error('AI returned no move');
                return;
            }
            
            // Step 2: Add line to the board
            const lineKey = getLineKey(aiLine.row1, aiLine.col1, aiLine.row2, aiLine.col2);
            room.board.lines[lineKey] = true;
            
            // Step 3: Check if a square was completed
            const completedSquares = checkForCompletedSquares(
                room.board, 
                aiLine.row1, 
                aiLine.col1, 
                aiLine.row2, 
                aiLine.col2
            );
            
            let extraTurn = false;
            
            if (completedSquares.length > 0) {
                // Award points to AI
                room.scores['ai-player'] += completedSquares.length;
                console.log('AI score updated:', room.scores['ai-player']); // Debug log
                
                // Mark squares as owned
                completedSquares.forEach(square => {
                    room.board.squares[square] = 'ai-player';
                });
                
                extraTurn = true;
            }
            
            // Step 4: Broadcast the AI move
            io.to(roomCode).emit('moveMade', {
                line: aiLine,
                player: 'ai-player',
                completedSquares,
                scores: room.scores
            });
            
            // Step 5: Update the AI with the results for learning
            await axios.post('http://localhost:5000/api/update', {
                board: room.board,
                completed_squares: completedSquares,
                player_id: 'ai-player'
            });
            
            // Check if game is over
            if (isGameOver(room.board)) {
                io.to(roomCode).emit('gameOver', { scores: room.scores });
                return;
            }
            
            // Make another move if AI gets an extra turn
            if (extraTurn) {
                makeAIMove(roomCode);
            } else {
                // Switch back to human player
                room.currentPlayer = room.players[0];
                // Include scores in turn change notification
                io.to(roomCode).emit('turnChange', { 
                    currentPlayer: room.currentPlayer,
                    scores: room.scores
                });
            }
        } catch (error) {
            console.error('Error making AI move:', error.message);
            // Fallback to random move if AI server fails
            makeRandomAIMove(roomCode);
        }
    }, 1000);
}

// Fallback function for random AI moves if AI server fails
function makeRandomAIMove(roomCode) {
    const room = gameRooms[roomCode];
    if (!room || !room.aiMode) return;
    
    const availableLines = findAvailableLines(room.board);
    
    if (availableLines.length > 0) {
        // Choose a random line
        const randomIndex = Math.floor(Math.random() * availableLines.length);
        const aiLine = availableLines[randomIndex];
        
        // Add line to the board
        const lineKey = getLineKey(aiLine.row1, aiLine.col1, aiLine.row2, aiLine.col2);
        room.board.lines[lineKey] = true;
        
        // Check if a square was completed
        const completedSquares = checkForCompletedSquares(
            room.board, 
            aiLine.row1, 
            aiLine.col1, 
            aiLine.row2, 
            aiLine.col2
        );
        
        let extraTurn = false;
        
        if (completedSquares.length > 0) {
            // Award points to AI
            room.scores['ai-player'] += completedSquares.length;
            console.log('AI score updated (random):', room.scores['ai-player']); // Debug log
            
            // Mark squares as owned
            completedSquares.forEach(square => {
                room.board.squares[square] = 'ai-player';
            });
            
            extraTurn = true;
        }
        
        // Broadcast the AI move
        io.to(roomCode).emit('moveMade', {
            line: aiLine,
            player: 'ai-player',
            completedSquares,
            scores: room.scores
        });
        
        // Check if game is over
        if (isGameOver(room.board)) {
            io.to(roomCode).emit('gameOver', { scores: room.scores });
            return;
        }
        
        // Make another move if AI gets an extra turn
        if (extraTurn) {
            makeRandomAIMove(roomCode);
        } else {
            // Switch back to human player
            room.currentPlayer = room.players[0];
            // Include scores in turn change notification
            io.to(roomCode).emit('turnChange', { 
                currentPlayer: room.currentPlayer,
                scores: room.scores
            });
        }
    }
}

// Find available lines on the board
function findAvailableLines(board) {
    const availableLines = [];
    const gridSize = 5; // 5x5 grid
    
    // Check horizontal lines
    for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize - 1; col++) {
            const lineKey = getLineKey(row, col, row, col + 1);
            if (!board.lines[lineKey]) {
                availableLines.push({
                    row1: row,
                    col1: col,
                    row2: row,
                    col2: col + 1
                });
            }
        }
    }
    
    // Check vertical lines
    for (let row = 0; row < gridSize - 1; row++) {
        for (let col = 0; col < gridSize; col++) {
            const lineKey = getLineKey(row, col, row + 1, col);
            if (!board.lines[lineKey]) {
                availableLines.push({
                    row1: row,
                    col1: col,
                    row2: row + 1,
                    col2: col
                });
            }
        }
    }
    
    return availableLines;
}

// Check if game is over (all possible squares formed)
function isGameOver(board) {
    const gridSize = 5; // 5x5 grid
    const totalPossibleSquares = (gridSize - 1) * (gridSize - 1);
    
    return Object.keys(board.squares).length >= totalPossibleSquares;
}

// Check for completed squares
function checkForCompletedSquares(board, row1, col1, row2, col2) {
    const completedSquares = [];
    
    // Determine if the line is horizontal or vertical
    const isHorizontal = row1 === row2;
    
    if (isHorizontal) {
        // Check square above
        if (row1 > 0) {
            const topLeft = `${row1-1},${Math.min(col1, col2)}`;
            const topRight = `${row1-1},${Math.max(col1, col2)}`;
            const top = getLineKey(row1-1, col1, row1-1, col2);
            const left = getLineKey(row1-1, Math.min(col1, col2), row1, Math.min(col1, col2));
            const right = getLineKey(row1-1, Math.max(col1, col2), row1, Math.max(col1, col2));
            
            if (board.lines[top] && board.lines[left] && board.lines[right]) {
                completedSquares.push(`${row1-1},${Math.min(col1, col2)}`);
            }
        }
        
        // Check square below
        if (row1 < 4) {
            const bottomLeft = `${row1},${Math.min(col1, col2)}`;
            const bottomRight = `${row1},${Math.max(col1, col2)}`;
            const bottom = getLineKey(row1+1, col1, row1+1, col2);
            const left = getLineKey(row1, Math.min(col1, col2), row1+1, Math.min(col1, col2));
            const right = getLineKey(row1, Math.max(col1, col2), row1+1, Math.max(col1, col2));
            
            if (board.lines[bottom] && board.lines[left] && board.lines[right]) {
                completedSquares.push(`${row1},${Math.min(col1, col2)}`);
            }
        }
    } else {
        // Check square to the left
        if (col1 > 0) {
            const topLeft = `${Math.min(row1, row2)},${col1-1}`;
            const bottomLeft = `${Math.max(row1, row2)},${col1-1}`;
            const left = getLineKey(row1, col1-1, row2, col1-1);
            const top = getLineKey(Math.min(row1, row2), col1-1, Math.min(row1, row2), col1);
            const bottom = getLineKey(Math.max(row1, row2), col1-1, Math.max(row1, row2), col1);
            
            if (board.lines[left] && board.lines[top] && board.lines[bottom]) {
                completedSquares.push(`${Math.min(row1, row2)},${col1-1}`);
            }
        }
        
        // Check square to the right
        if (col1 < 4) {
            const topRight = `${Math.min(row1, row2)},${col1}`;
            const bottomRight = `${Math.max(row1, row2)},${col1}`;
            const right = getLineKey(row1, col1+1, row2, col1+1);
            const top = getLineKey(Math.min(row1, row2), col1, Math.min(row1, row2), col1+1);
            const bottom = getLineKey(Math.max(row1, row2), col1, Math.max(row1, row2), col1+1);
            
            if (board.lines[right] && board.lines[top] && board.lines[bottom]) {
                completedSquares.push(`${Math.min(row1, row2)},${col1}`);
            }
        }
    }
    
    return completedSquares;
}

// Generate a unique room code
function generateRoomCode() {
    return Math.random().toString(36).substring(2, 7).toUpperCase();
}

// Create a unique key for each line
function getLineKey(row1, col1, row2, col2) {
    return `${Math.min(row1, row2)},${Math.min(col1, col2)}-${Math.max(row1, row2)},${Math.max(col1, col2)}`;
}

// Create an empty game board
function createEmptyBoard() {
    return {
        lines: {},
        squares: {}
    };
}

const PORT = process.env.PORT || 3000;
server.listen(PORT, () => {
    console.log(`Server is running on port ${PORT}`);
}); 