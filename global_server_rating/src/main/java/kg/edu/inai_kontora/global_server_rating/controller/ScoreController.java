package kg.edu.inai_kontora.global_server_rating.controller;

import kg.edu.inai_kontora.global_server_rating.dto.ScoreEntry;
import org.springframework.http.HttpStatus;
import org.springframework.http.ResponseEntity;
import org.springframework.web.bind.annotation.*;

import java.time.LocalDateTime;
import java.util.Comparator;
import java.util.List;
import java.util.Map;
import java.util.concurrent.ConcurrentHashMap;
import java.util.stream.Collectors;

@RestController
public class ScoreController {

    // Потокобезопасное хранилище результатов в памяти
    // Ключ: teamId, Значение: ScoreEntry
    private final Map<String, ScoreEntry> teamScores = new ConcurrentHashMap<>();

    @PostMapping("/submit_score")
    public ResponseEntity<String> submitScore(@RequestBody ScoreEntry scoreSubmission) {
        if (scoreSubmission.getTeamId() == null || scoreSubmission.getTeamId().trim().isEmpty()) {
            return ResponseEntity.badRequest().body("team_id не может быть пустым");
        }

        // Используем scoreSubmission напрямую или создаем новый ScoreEntry для контроля времени
        ScoreEntry currentEntry = teamScores.get(scoreSubmission.getTeamId());
        if (currentEntry != null) {
            currentEntry.setAccuracy(scoreSubmission.getAccuracy());
            currentEntry.recordUpdate(); // Обновить время
        } else {
            // Если новая команда, создаем новую запись с текущим временем
            currentEntry = new ScoreEntry(scoreSubmission.getTeamId(), scoreSubmission.getAccuracy());
        }
        teamScores.put(scoreSubmission.getTeamId(), currentEntry);

        System.out.println("[" + LocalDateTime.now() + "] Получен результат: Команда " +
                currentEntry.getTeamId() + ", Точность " + String.format("%.2f", currentEntry.getAccuracy()) + "%");
        return ResponseEntity.ok("{\"message\": \"Score received successfully\"}");
    }

    @GetMapping("/leaderboard_data")
    public ResponseEntity<List<ScoreEntry>> getLeaderboardData() {
        List<ScoreEntry> sortedScores = teamScores.values().stream()
                .sorted(Comparator.comparingDouble(ScoreEntry::getAccuracy).reversed() // Сначала по точности (убывание)
                        .thenComparing(ScoreEntry::getLastUpdateTime).reversed())      // Потом по времени (новейшие выше)
                .collect(Collectors.toList());
        return ResponseEntity.ok(sortedScores);
    }

    // Если вы хотите, чтобы Spring Boot обслуживал leaderboard.html из static/
    // этот маршрут не обязателен, если leaderboard.html - это index.html в static/
    // или если вы настроили view resolver для Thymeleaf, например.
    // Для простоты, если leaderboard.html лежит в static/, Spring Boot отдаст его по запросу /leaderboard.html
    // Чтобы он отдавался по /, можно сделать так:
    // @GetMapping("/")
    // public String home() {
    //     return "redirect:/leaderboard.html";
    // }
}