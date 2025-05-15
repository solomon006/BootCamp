package kg.edu.inai_kontora.global_server_rating.dto;

import com.fasterxml.jackson.annotation.JsonProperty;
import java.time.LocalDateTime;
import java.time.format.DateTimeFormatter;

public class ScoreEntry {
    @JsonProperty("team_id")
    private String teamId;
    private double accuracy;
    private String lastUpdateTime;

    // Конструктор по умолчанию нужен для десериализации JSON
    public ScoreEntry() {
    }

    public ScoreEntry(String teamId, double accuracy) {
        this.teamId = teamId;
        this.accuracy = accuracy;
        this.lastUpdateTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
    }

    // Геттеры
    public String getTeamId() {
        return teamId;
    }

    public double getAccuracy() {
        return accuracy;
    }

    public String getLastUpdateTime() {
        return lastUpdateTime;
    }

    // Сеттеры (могут понадобиться для Jackson при POST запросах)
    public void setTeamId(String teamId) {
        this.teamId = teamId;
    }

    public void setAccuracy(double accuracy) {
        this.accuracy = accuracy;
    }

    public void setLastUpdateTime(String lastUpdateTime) {
        // Обычно lastUpdateTime устанавливается при создании/обновлении,
        // но сеттер может быть нужен для некоторых сценариев десериализации
        this.lastUpdateTime = lastUpdateTime;
    }

    // Для удобного обновления времени при перезаписи счета
    public void recordUpdate() {
        this.lastUpdateTime = LocalDateTime.now().format(DateTimeFormatter.ofPattern("yyyy-MM-dd HH:mm:ss"));
    }
}