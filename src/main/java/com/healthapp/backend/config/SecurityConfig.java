package com.healthapp.backend.config;

import org.springframework.context.annotation.Bean;
import org.springframework.context.annotation.Configuration;
import org.springframework.security.config.annotation.web.builders.HttpSecurity;
import org.springframework.security.web.SecurityFilterChain;

@Configuration
public class SecurityConfig {

    @Bean
    public SecurityFilterChain securityFilterChain(HttpSecurity http) throws Exception {
        http
                .csrf(csrf -> csrf.disable()) // disable CSRF for simplicity (enable later for production)
                .authorizeHttpRequests(auth -> auth
                        .requestMatchers("/api/**").permitAll() // allow our test endpoint
                        .anyRequest().authenticated()
                );
        return http.build();
    }
}