
const RestartButton: React.FC = () => {
    const ip_and_port = 'put_urs_here_dont_forget_the_slash->/';
    const onClickRestart = async () => {
        try { 
            const response = await fetch(ip_and_port + 'new_game');
          } catch (error) {
            console.error('Error fetching game state:', error);
          }
    };

    return(<button onClick={onClickRestart}>restart</button>);
}
export default RestartButton;
